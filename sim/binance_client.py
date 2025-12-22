"""
Binance Data Client - Obtención de datos reales de mercado

NORMA DURA:
- Solo OBTIENE datos, no ejecuta trades
- Transforma datos a dimensiones estructurales abstractas
- NO genera señales de compra/venta
- Los datos se usan para simulación in silico

Dimensiones abstractas derivadas:
- liquidity: Volumen 24h normalizado
- volatility: Variación de precio (high-low)/price
- concentration: Ratio de volumen top coins
- latency: Inverso de frecuencia de trades
- sentiment: Cambio de precio 24h normalizado
- leverage: Open interest / volume (si disponible)
- network_load: Número de trades normalizado
- regulatory_pressure: Estático (no disponible via API)
"""

import hmac
import hashlib
import time
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketData:
    """Datos de mercado de un símbolo."""
    symbol: str
    price: float
    price_change_24h: float      # Porcentaje
    high_24h: float
    low_24h: float
    volume_24h: float            # En quote currency (USDT)
    trades_24h: int
    timestamp: datetime


@dataclass
class AbstractState:
    """Estado abstracto derivado de datos reales."""
    liquidity: float        # 0-1
    volatility: float       # 0-1
    concentration: float    # 0-1
    latency: float          # 0-1
    sentiment: float        # 0-1
    leverage: float         # 0-1
    network_load: float     # 0-1
    regulatory_pressure: float  # 0-1

    source_symbols: List[str]
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            'liquidity': self.liquidity,
            'volatility': self.volatility,
            'concentration': self.concentration,
            'latency': self.latency,
            'sentiment': self.sentiment,
            'leverage': self.leverage,
            'network_load': self.network_load,
            'regulatory_pressure': self.regulatory_pressure,
            'source': 'binance',
            'symbols': self.source_symbols,
            'timestamp': self.timestamp.isoformat()
        }


class BinanceClient:
    """
    Cliente Binance para datos de mercado.

    ⚠️ Solo lectura de datos. NO ejecuta trades.
    """

    BASE_URL = "https://api.binance.com"

    # Símbolos principales para análisis
    DEFAULT_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
        'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT'
    ]

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={'X-MBX-APIKEY': self.api_key}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _sign(self, params: Dict) -> str:
        """Firma los parámetros para endpoints autenticados."""
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _request(self, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Realiza petición a la API."""
        session = await self._get_session()

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign(params)

        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                raise Exception(f"Binance API error {response.status}: {text}")

    # ==========================================================================
    # ENDPOINTS PÚBLICOS (no requieren firma)
    # ==========================================================================

    async def get_ticker_24h(self, symbol: str) -> MarketData:
        """Obtiene datos de 24h para un símbolo."""
        data = await self._request('/api/v3/ticker/24hr', {'symbol': symbol})

        return MarketData(
            symbol=symbol,
            price=float(data['lastPrice']),
            price_change_24h=float(data['priceChangePercent']),
            high_24h=float(data['highPrice']),
            low_24h=float(data['lowPrice']),
            volume_24h=float(data['quoteVolume']),
            trades_24h=int(data['count']),
            timestamp=datetime.now()
        )

    async def get_all_tickers(self, symbols: Optional[List[str]] = None) -> List[MarketData]:
        """Obtiene datos de múltiples símbolos."""
        symbols = symbols or self.DEFAULT_SYMBOLS

        tasks = [self.get_ticker_24h(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrar errores
        return [r for r in results if isinstance(r, MarketData)]

    async def get_price(self, symbol: str) -> float:
        """Obtiene precio actual."""
        data = await self._request('/api/v3/ticker/price', {'symbol': symbol})
        return float(data['price'])

    async def get_orderbook_depth(self, symbol: str, limit: int = 100) -> Dict:
        """Obtiene profundidad del orderbook."""
        data = await self._request('/api/v3/depth', {'symbol': symbol, 'limit': limit})

        bids_volume = sum(float(b[1]) for b in data['bids'])
        asks_volume = sum(float(a[1]) for a in data['asks'])

        return {
            'symbol': symbol,
            'bids_volume': bids_volume,
            'asks_volume': asks_volume,
            'ratio': bids_volume / asks_volume if asks_volume > 0 else 1.0,
            'spread': float(data['asks'][0][0]) - float(data['bids'][0][0]) if data['asks'] and data['bids'] else 0
        }

    async def ping(self) -> bool:
        """Verifica conexión con Binance."""
        try:
            await self._request('/api/v3/ping')
            return True
        except:
            return False

    # ==========================================================================
    # TRANSFORMACIÓN A ESTADO ABSTRACTO
    # ==========================================================================

    async def get_abstract_state(self, symbols: Optional[List[str]] = None) -> AbstractState:
        """
        Obtiene estado abstracto del mercado basado en datos reales.

        Transforma métricas de mercado a dimensiones estructurales 0-1.
        """
        symbols = symbols or self.DEFAULT_SYMBOLS

        # Obtener datos
        tickers = await self.get_all_tickers(symbols)

        if not tickers:
            raise Exception("No se pudieron obtener datos de mercado")

        # Calcular dimensiones abstractas

        # 1. LIQUIDEZ: Volumen total normalizado (log scale)
        total_volume = sum(t.volume_24h for t in tickers)
        # $1B = 0.5, $10B = 0.7, $100B = 0.9
        liquidity = min(1.0, max(0.1, (total_volume / 1e10) ** 0.3))

        # 2. VOLATILIDAD: Promedio de (high-low)/price
        volatilities = []
        for t in tickers:
            if t.price > 0:
                vol = (t.high_24h - t.low_24h) / t.price
                volatilities.append(vol)
        avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0.05
        # Normalizar: 2% = 0.2, 5% = 0.5, 10% = 0.8
        volatility = min(1.0, avg_volatility * 10)

        # 3. CONCENTRACIÓN: Ratio del top 2 vs resto
        volumes = sorted([t.volume_24h for t in tickers], reverse=True)
        if len(volumes) >= 3 and sum(volumes) > 0:
            top2_ratio = sum(volumes[:2]) / sum(volumes)
            concentration = top2_ratio  # Ya está 0-1
        else:
            concentration = 0.5

        # 4. LATENCIA: Inverso de trades (más trades = menor latencia)
        total_trades = sum(t.trades_24h for t in tickers)
        # 1M trades = 0.3, 10M = 0.2, 100M = 0.1
        latency = max(0.05, min(0.8, 1.0 / (1 + total_trades / 1e7)))

        # 5. SENTIMENT: Promedio de cambio 24h normalizado
        changes = [t.price_change_24h for t in tickers]
        avg_change = sum(changes) / len(changes) if changes else 0
        # -10% = 0.2, 0% = 0.5, +10% = 0.8
        sentiment = max(0.1, min(0.9, 0.5 + avg_change / 20))

        # 6. LEVERAGE: Aproximación basada en volatilidad + volumen
        # (Real leverage requiere datos de futuros)
        leverage = min(0.8, volatility * 0.6 + (1 - liquidity) * 0.4)

        # 7. NETWORK_LOAD: Basado en número de trades
        # 10M trades/día = 0.5
        network_load = min(0.9, total_trades / 2e7)

        # 8. REGULATORY_PRESSURE: No disponible via API, usar valor neutral
        regulatory_pressure = 0.4  # Valor base

        return AbstractState(
            liquidity=round(liquidity, 3),
            volatility=round(volatility, 3),
            concentration=round(concentration, 3),
            latency=round(latency, 3),
            sentiment=round(sentiment, 3),
            leverage=round(leverage, 3),
            network_load=round(network_load, 3),
            regulatory_pressure=round(regulatory_pressure, 3),
            source_symbols=[t.symbol for t in tickers],
            timestamp=datetime.now()
        )

    async def get_symbol_state(self, symbol: str) -> AbstractState:
        """Estado abstracto para un símbolo específico."""
        ticker = await self.get_ticker_24h(symbol)
        depth = await self.get_orderbook_depth(symbol, limit=50)

        # Volatilidad del símbolo
        volatility = (ticker.high_24h - ticker.low_24h) / ticker.price if ticker.price > 0 else 0.05
        volatility = min(1.0, volatility * 10)

        # Liquidez basada en volumen
        liquidity = min(1.0, (ticker.volume_24h / 1e9) ** 0.4)

        # Sentiment basado en cambio de precio
        sentiment = max(0.1, min(0.9, 0.5 + ticker.price_change_24h / 20))

        # Concentración basada en ratio bid/ask
        concentration = abs(depth['ratio'] - 1) / 2  # Cuanto más desbalanceado, más concentrado
        concentration = min(0.9, concentration)

        # Latencia inversa a trades
        latency = max(0.05, min(0.8, 1.0 / (1 + ticker.trades_24h / 1e6)))

        return AbstractState(
            liquidity=round(liquidity, 3),
            volatility=round(volatility, 3),
            concentration=round(concentration, 3),
            latency=round(latency, 3),
            sentiment=round(sentiment, 3),
            leverage=round(volatility * 0.5, 3),
            network_load=round(min(0.9, ticker.trades_24h / 5e6), 3),
            regulatory_pressure=0.4,
            source_symbols=[symbol],
            timestamp=datetime.now()
        )


# Configuración
BINANCE_API_KEY = "xSJwtN2YqTa8ONYblN1iLrpU72bthf0bYg0hGWOPD1THmgPSgiX2Epqy0qaKJ1fJ"
BINANCE_API_SECRET = "owWXUN3Zc8rr4Z1cVc4SOA2sGwjws0ouqCdGWp2XhNjwKKiCNo3wQqgvu3h3uAe3"

# Singleton
_binance_client: Optional[BinanceClient] = None

def get_binance_client() -> BinanceClient:
    global _binance_client
    if _binance_client is None:
        _binance_client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
    return _binance_client
