#!/bin/bash
# Iniciar el Mundo Vivo como daemon

cd /root/NEO_EVA

# Verificar si ya está corriendo
if pgrep -f "living_world_daemon.py" > /dev/null; then
    echo "El Mundo Vivo ya está corriendo."
    echo "Para detenerlo: ./stop_living_world.sh"
    exit 1
fi

# Crear directorio de logs
mkdir -p logs

# Iniciar en background
echo "Iniciando Mundo Vivo..."
nohup python3 worlds/living_world_daemon.py > logs/living_world_daemon.out 2>&1 &

PID=$!
echo $PID > /tmp/living_world.pid

echo "Mundo Vivo iniciado (PID: $PID)"
echo "Log: tail -f logs/living_world.log"
echo "Output: tail -f logs/living_world_daemon.out"
echo "Para detener: ./stop_living_world.sh"
