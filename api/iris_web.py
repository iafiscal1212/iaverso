#!/usr/bin/env python3
"""
🔮 IRIS Web - Interfaz tipo ChatGPT con persistencia y ejecucion autonoma
"""

import sys
import os
import json
import uuid
from datetime import datetime
sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/core')
sys.path.insert(0, '/root/NEO_EVA/api')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import asyncio

from iris_autonomo import (
    generar_codigo, revisar_codigo, explicar_codigo,
    resolver_problema, buscar_web, investigar_tema,
    pensar, cargar_autonomia,
    # Funciones de seguridad
    escanear_seguridad, analizar_codigo_seguro, escaneo_completo_sistema,
    obtener_reporte_seguridad, verificar_archivo_seguro,
    # Funciones de auto-curacion
    autodiagnostico, detectar_y_arreglar, verificar_salud
)

# Sistema de ejecucion y aprobaciones
from iris_executor import IrisExecutor, RiskLevel
from iris_approval_queue import (
    ApprovalQueue, ActionType, PendingAction,
    crear_accion_ejecutar_python, crear_accion_ejecutar_bash,
    crear_accion_escribir_archivo
)

# Instancias globales
executor = IrisExecutor()
approval_queue = ApprovalQueue()

app = FastAPI(title="IRIS Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== WebSocket Manager ==========
class ConnectionManager:
    """Gestiona conexiones WebSocket para actualizaciones en tiempo real"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Envia mensaje a todos los clientes conectados"""
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)


ws_manager = ConnectionManager()


async def notificar_nueva_accion(action: dict):
    """Notifica a todos los clientes de una nueva accion pendiente"""
    await ws_manager.broadcast({
        "type": "nueva_accion",
        "action": action
    })


async def notificar_accion_completada(action_id: str, resultado: dict):
    """Notifica que una accion fue completada"""
    await ws_manager.broadcast({
        "type": "accion_completada",
        "action_id": action_id,
        "resultado": resultado
    })


# Directorio para guardar conversaciones
CONVERSACIONES_DIR = "/root/NEO_EVA/data/iris_conversaciones"
os.makedirs(CONVERSACIONES_DIR, exist_ok=True)


def guardar_conversacion(conv_id: str, datos: dict):
    filepath = os.path.join(CONVERSACIONES_DIR, f"{conv_id}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


def cargar_conversacion(conv_id: str) -> dict:
    filepath = os.path.join(CONVERSACIONES_DIR, f"{conv_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def listar_conversaciones() -> List[dict]:
    conversaciones = []
    for filename in os.listdir(CONVERSACIONES_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(CONVERSACIONES_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversaciones.append({
                        'id': filename.replace('.json', ''),
                        'titulo': data.get('titulo', 'Sin título'),
                        'fecha': data.get('fecha_creacion', ''),
                        'mensajes': len(data.get('mensajes', []))
                    })
            except:
                pass
    conversaciones.sort(key=lambda x: x['fecha'], reverse=True)
    return conversaciones


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔮 IRIS</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #fff;
            height: 100vh;
            display: flex;
        }
        /* Sidebar */
        .sidebar {
            width: 260px;
            background: #111827;
            border-right: 1px solid #333;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid #333;
        }
        .new-chat-btn {
            width: 100%;
            padding: 12px;
            background: #4f46e5;
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            justify-content: center;
        }
        .new-chat-btn:hover { background: #4338ca; }
        .conversations-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }
        .conv-item {
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .conv-item:hover { background: #1f2937; }
        .conv-item.active { background: #374151; }
        .conv-title {
            font-size: 13px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex: 1;
        }
        .conv-actions {
            display: none;
            gap: 4px;
        }
        .conv-item:hover .conv-actions { display: flex; }
        .conv-action {
            padding: 4px 8px;
            background: none;
            border: none;
            color: #9ca3af;
            cursor: pointer;
            font-size: 12px;
        }
        .conv-action:hover { color: #fff; }
        .conv-action.delete:hover { color: #ef4444; }
        /* Main area */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .header {
            padding: 16px 24px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 1.4em;
            background: linear-gradient(90deg, #a855f7, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }
        .message {
            margin-bottom: 24px;
            display: flex;
            gap: 16px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        .message.user { flex-direction: row-reverse; }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }
        .message.iris .avatar { background: linear-gradient(135deg, #a855f7, #6366f1); }
        .message.user .avatar { background: #374151; }
        .content {
            flex: 1;
            padding: 16px;
            border-radius: 12px;
            line-height: 1.6;
            font-size: 15px;
        }
        .message.iris .content { background: #1f2937; }
        .message.user .content { background: #4f46e5; }
        .content pre {
            background: #0d1117;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 12px 0;
            font-size: 13px;
        }
        .content code { font-family: 'Fira Code', 'Consolas', monospace; }
        /* Input area */
        .input-area {
            padding: 16px 24px;
            border-top: 1px solid #333;
            background: #111827;
        }
        .input-wrapper {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        textarea {
            flex: 1;
            padding: 14px 16px;
            border: 1px solid #333;
            border-radius: 12px;
            background: #1f2937;
            color: #fff;
            font-size: 15px;
            resize: none;
            outline: none;
            min-height: 52px;
            max-height: 200px;
            font-family: inherit;
        }
        textarea:focus { border-color: #6366f1; }
        .send-btn {
            padding: 14px 24px;
            background: linear-gradient(135deg, #a855f7, #6366f1);
            border: none;
            border-radius: 12px;
            color: white;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }
        .send-btn:hover { transform: scale(1.02); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .upload-btn {
            padding: 14px;
            background: #374151;
            border: none;
            border-radius: 12px;
            color: #fff;
            cursor: pointer;
            font-size: 18px;
        }
        .file-input { display: none; }
        .file-preview {
            max-width: 800px;
            margin: 0 auto 12px;
            background: #374151;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .file-preview .remove { cursor: pointer; color: #ef4444; }
        .loading {
            display: flex;
            gap: 4px;
            padding: 16px;
        }
        .loading span {
            width: 8px;
            height: 8px;
            background: #6366f1;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        .loading span:nth-child(1) { animation-delay: -0.32s; }
        .loading span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        /* Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            justify-content: center;
            align-items: center;
            z-index: 100;
        }
        .modal.show { display: flex; }
        .modal-content {
            background: #1f2937;
            padding: 24px;
            border-radius: 12px;
            width: 400px;
        }
        .modal-content h3 { margin-bottom: 16px; }
        .modal-content input {
            width: 100%;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 8px;
            background: #111827;
            color: #fff;
            margin-bottom: 16px;
        }
        .modal-buttons { display: flex; gap: 8px; justify-content: flex-end; }
        .modal-buttons button {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .modal-buttons .cancel { background: #374151; color: #fff; }
        .modal-buttons .save { background: #4f46e5; color: #fff; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #6b7280;
        }
        .empty-state h2 { font-size: 1.5em; margin-bottom: 8px; color: #9ca3af; }
        /* Panel de acciones pendientes */
        .pending-panel {
            border-top: 1px solid #333;
            padding: 12px;
            background: #0f172a;
        }
        .pending-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .pending-header h4 {
            font-size: 13px;
            color: #f59e0b;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .pending-count {
            background: #ef4444;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
        }
        .pending-action {
            background: #1e293b;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            border-left: 3px solid #f59e0b;
        }
        .pending-action.high { border-left-color: #ef4444; }
        .pending-action.critical { border-left-color: #dc2626; background: #1c1917; }
        .pending-action .desc {
            font-size: 12px;
            color: #e2e8f0;
            margin-bottom: 6px;
        }
        .pending-action .risk {
            font-size: 10px;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        .pending-action .buttons {
            display: flex;
            gap: 6px;
        }
        .pending-action .btn-approve {
            flex: 1;
            padding: 6px;
            background: #22c55e;
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            font-size: 11px;
        }
        .pending-action .btn-reject {
            flex: 1;
            padding: 6px;
            background: #ef4444;
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            font-size: 11px;
        }
        .pending-action .btn-approve:hover { background: #16a34a; }
        .pending-action .btn-reject:hover { background: #dc2626; }
        .no-pending {
            text-align: center;
            color: #6b7280;
            font-size: 12px;
            padding: 12px;
        }
        /* Notificacion flotante */
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #1e293b;
            border: 1px solid #f59e0b;
            border-radius: 12px;
            padding: 16px 20px;
            z-index: 1000;
            display: none;
            animation: slideIn 0.3s ease;
            max-width: 350px;
        }
        .notification.show { display: block; }
        .notification h4 {
            color: #f59e0b;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .notification p {
            color: #e2e8f0;
            font-size: 13px;
            margin-bottom: 12px;
        }
        .notification .buttons {
            display: flex;
            gap: 8px;
        }
        .notification button {
            flex: 1;
            padding: 8px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
        }
        .notification .btn-yes { background: #22c55e; color: white; }
        .notification .btn-no { background: #6b7280; color: white; }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        /* Resultado de ejecucion */
        .execution-result {
            background: #0d1117;
            border-radius: 8px;
            padding: 12px;
            margin: 12px 0;
            border-left: 3px solid #22c55e;
        }
        .execution-result.error { border-left-color: #ef4444; }
        .execution-result pre {
            margin: 0;
            font-size: 12px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <button class="new-chat-btn" onclick="newChat()">
                <span>+</span> Nueva conversación
            </button>
        </div>
        <div class="conversations-list" id="convList"></div>

        <!-- Panel de acciones pendientes -->
        <div class="pending-panel" id="pendingPanel">
            <div class="pending-header">
                <h4>⚡ Acciones Pendientes</h4>
                <span class="pending-count" id="pendingCount">0</span>
            </div>
            <div id="pendingList">
                <div class="no-pending">No hay acciones pendientes</div>
            </div>
        </div>
    </div>

    <!-- Notificacion flotante -->
    <div class="notification" id="notification">
        <h4 id="notifTitle">🔮 ¡Hola! Necesito tu permiso</h4>
        <p id="notifDesc">Descripcion de la accion</p>
        <div class="buttons">
            <button class="btn-yes" onclick="approveFromNotif()">✅ Si, adelante</button>
            <button class="btn-no" onclick="hideNotification()">❌ Ahora no</button>
        </div>
    </div>

    <div class="main">
        <div class="header">
            <h1>🔮 IRIS</h1>
            <span id="convTitle" style="color: #6b7280; font-size: 14px;"></span>
        </div>

        <div class="chat-container" id="chat">
            <div class="empty-state">
                <h2>🔮 IRIS</h2>
                <p>Tu asistente de código con IA</p>
                <p style="margin-top: 20px;">Crea una nueva conversación para empezar</p>
            </div>
        </div>

        <div class="input-area">
            <div id="filePreview"></div>
            <div class="input-wrapper">
                <input type="file" id="fileInput" class="file-input" onchange="handleFile(this)" accept=".py,.js,.ts,.json,.txt,.md,.html,.css,.sql,.go,.rs">
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">📎</button>
                <textarea id="input" placeholder="Escribe tu mensaje..." rows="1" onkeydown="handleKey(event)"></textarea>
                <button class="send-btn" id="sendBtn" onclick="send()">Enviar</button>
            </div>
        </div>
    </div>

    <!-- Modal para renombrar -->
    <div class="modal" id="renameModal">
        <div class="modal-content">
            <h3>Renombrar conversación</h3>
            <input type="text" id="renameInput" placeholder="Nuevo título...">
            <div class="modal-buttons">
                <button class="cancel" onclick="closeModal()">Cancelar</button>
                <button class="save" onclick="saveRename()">Guardar</button>
            </div>
        </div>
    </div>

    <script>
        let currentConvId = null;
        let uploadedFile = null;
        let conversations = {};

        // Cargar conversaciones al inicio
        loadConversations();

        async function loadConversations() {
            try {
                const resp = await fetch('/conversaciones');
                const data = await resp.json();
                renderConvList(data);
            } catch(e) {
                console.error('Error cargando conversaciones:', e);
            }
        }

        function renderConvList(convs) {
            const list = document.getElementById('convList');
            if (convs.length === 0) {
                list.innerHTML = '<div style="padding: 20px; text-align: center; color: #6b7280; font-size: 13px;">No hay conversaciones</div>';
                return;
            }
            list.innerHTML = convs.map(c => `
                <div class="conv-item ${c.id === currentConvId ? 'active' : ''}" onclick="loadConv('${c.id}')">
                    <span class="conv-title">💬 ${c.titulo}</span>
                    <div class="conv-actions">
                        <button class="conv-action" onclick="event.stopPropagation(); renameConv('${c.id}', '${c.titulo}')">✏️</button>
                        <button class="conv-action delete" onclick="event.stopPropagation(); deleteConv('${c.id}')">🗑️</button>
                    </div>
                </div>
            `).join('');
        }

        async function newChat() {
            currentConvId = null;
            document.getElementById('chat').innerHTML = `
                <div class="message iris">
                    <div class="avatar">🔮</div>
                    <div class="content">
                        ¡Hola! Soy <strong>IRIS</strong>, tu asistente de programación creativa.<br><br>
                        Puedo ayudarte a:<br>
                        • Diseñar y crear aplicaciones completas<br>
                        • Resolver problemas técnicos<br>
                        • Revisar y mejorar tu código<br>
                        • Buscar información actualizada<br><br>
                        ¿Qué quieres crear hoy?
                    </div>
                </div>
            `;
            document.getElementById('convTitle').textContent = 'Nueva conversación';
            loadConversations();
        }

        async function loadConv(id) {
            try {
                const resp = await fetch('/conversacion/' + id);
                const data = await resp.json();
                currentConvId = id;
                document.getElementById('convTitle').textContent = data.titulo;

                const chat = document.getElementById('chat');
                chat.innerHTML = data.mensajes.map(m => `
                    <div class="message ${m.rol}">
                        <div class="avatar">${m.rol === 'user' ? '👤' : '🔮'}</div>
                        <div class="content">${formatMessage(m.contenido)}</div>
                    </div>
                `).join('');
                chat.scrollTop = chat.scrollHeight;
                loadConversations();
            } catch(e) {
                console.error('Error cargando conversación:', e);
            }
        }

        async function deleteConv(id) {
            if (confirm('¿Eliminar esta conversación?')) {
                await fetch('/conversacion/' + id, { method: 'DELETE' });
                if (currentConvId === id) newChat();
                loadConversations();
            }
        }

        function renameConv(id, currentTitle) {
            document.getElementById('renameInput').value = currentTitle;
            document.getElementById('renameModal').classList.add('show');
            document.getElementById('renameModal').dataset.convId = id;
        }

        function closeModal() {
            document.getElementById('renameModal').classList.remove('show');
        }

        async function saveRename() {
            const id = document.getElementById('renameModal').dataset.convId;
            const newTitle = document.getElementById('renameInput').value.trim();
            if (newTitle) {
                await fetch('/conversacion/' + id + '/titulo', {
                    method: 'PUT',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({titulo: newTitle})
                });
                loadConversations();
                if (currentConvId === id) {
                    document.getElementById('convTitle').textContent = newTitle;
                }
            }
            closeModal();
        }

        function formatMessage(text) {
            return text
                .replace(/```(\\w*)\\n?([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\\n/g, '<br>');
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
            }
        }

        function handleFile(input) {
            const file = input.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedFile = { name: file.name, content: e.target.result };
                    document.getElementById('filePreview').innerHTML = `
                        <div class="file-preview">
                            📄 ${file.name}
                            <span class="remove" onclick="removeFile()">✕</span>
                        </div>
                    `;
                };
                reader.readAsText(file);
            }
        }

        function removeFile() {
            uploadedFile = null;
            document.getElementById('filePreview').innerHTML = '';
            document.getElementById('fileInput').value = '';
        }

        function addMessage(text, isUser) {
            const chat = document.getElementById('chat');
            // Remove empty state if exists
            const emptyState = chat.querySelector('.empty-state');
            if (emptyState) emptyState.remove();

            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'iris');
            div.innerHTML = `
                <div class="avatar">${isUser ? '👤' : '🔮'}</div>
                <div class="content">${formatMessage(text)}</div>
            `;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        // IRIS presenta una idea en el chat conversacionalmente
        function addIrisIdea(idea) {
            const chat = document.getElementById('chat');
            const emptyState = chat.querySelector('.empty-state');
            if (emptyState) emptyState.remove();

            const div = document.createElement('div');
            div.className = 'message iris idea-message';
            div.id = 'idea-' + idea.id;
            div.innerHTML = `
                <div class="avatar">🔮</div>
                <div class="content">
                    <div style="margin-bottom:12px;">
                        <strong>¡Oye! Estuve pensando...</strong>
                    </div>
                    <div style="background:#1e293b;padding:12px;border-radius:8px;margin-bottom:12px;">
                        <div style="font-size:15px;margin-bottom:8px;">💡 <strong>${idea.descripcion || idea.description}</strong></div>
                        <div style="color:#94a3b8;font-size:13px;">🤔 ${idea.razon || idea.context?.reason || ''}</div>
                    </div>
                    <div style="background:#0d1117;padding:10px;border-radius:6px;font-family:monospace;font-size:11px;max-height:150px;overflow:auto;margin-bottom:12px;">
                        <code>${(idea.codigo || idea.payload?.content || '').substring(0,400).replace(/</g,'&lt;')}...</code>
                    </div>
                    <div style="margin-bottom:8px;color:#cbd5e1;">¿Qué te parece? ¿Lo creo?</div>
                    <div style="display:flex;gap:10px;">
                        <button onclick="respondToIdea('${idea.id}', true)" style="background:#22c55e;color:white;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-weight:bold;">✅ Sí, hazlo</button>
                        <button onclick="respondToIdea('${idea.id}', false)" style="background:#64748b;color:white;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;">❌ No, piensa otra cosa</button>
                    </div>
                </div>
            `;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        // Responder a una idea de IRIS
        async function respondToIdea(ideaId, approved) {
            const ideaDiv = document.getElementById('idea-' + ideaId);

            if (approved) {
                // Mostrar que está trabajando
                if (ideaDiv) {
                    const buttons = ideaDiv.querySelector('div[style*="display:flex"]');
                    if (buttons) buttons.innerHTML = '<span style="color:#22c55e;">⏳ Creando...</span>';
                }

                try {
                    // Mostrar que está trabajando
                    addMessage('⏳ **Trabajando...**\\n\\nCreando el archivo...', false);

                    const resp = await fetch('/pending-actions/' + ideaId + '/approve', {method: 'POST'});
                    const data = await resp.json();

                    // Actualizar el mensaje con el resultado DETALLADO
                    if (ideaDiv) {
                        const buttons = ideaDiv.querySelector('div[style*="display:flex"]');
                        if (data.ok && data.result?.success) {
                            buttons.innerHTML = '<span style="color:#22c55e;">✅ ¡Hecho!</span>';

                            // Mostrar todo lo que hizo
                            let detalle = '✅ **¡Listo!**\\n\\n';
                            detalle += '📁 **Archivo creado:** `' + (data.result?.path || '') + '`\\n\\n';
                            detalle += '📊 **Tamaño:** ' + (data.result?.size || 0) + ' bytes\\n\\n';

                            if (data.result?.content) {
                                detalle += '💻 **Código guardado:**\\n```python\\n' + data.result.content.substring(0, 800) + '\\n```\\n\\n';
                            }

                            detalle += '🚀 **Para ejecutarlo:**\\n```bash\\npython3 ' + (data.result?.path || '') + '\\n```';

                            addMessage(detalle, false);
                        } else {
                            buttons.innerHTML = '<span style="color:#ef4444;">❌ Error</span>';
                            addMessage('❌ **Error al crear:**\\n\\n' + (data.result?.error || 'Error desconocido'), false);
                        }
                    }
                } catch(e) {
                    addMessage('❌ **Error:** ' + e.message, false);
                }
            } else {
                // Rechazado
                if (ideaDiv) {
                    const buttons = ideaDiv.querySelector('div[style*="display:flex"]');
                    if (buttons) buttons.innerHTML = '<span style="color:#94a3b8;">Vale, pensaré en otra cosa.</span>';
                }
                await fetch('/pending-actions/' + ideaId + '/reject', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({reason: 'Usuario prefiere otra idea'})
                });
                addMessage('Entendido, voy a pensar en otra cosa...', false);
            }
            loadPendingActions();
        }

        // Verificar si hay nuevos proyectos de IRIS cada 15 segundos
        let shownIdeaIds = new Set();
        setInterval(async () => {
            try {
                const resp = await fetch('/pending-actions');
                const data = await resp.json();
                const actions = data.actions || data;

                for (const action of actions) {
                    const source = action.context?.source || '';
                    const esProyectoIris = source.includes('iris_pensamiento') || source.includes('iris_autonoma');

                    if (!shownIdeaIds.has(action.id) && esProyectoIris) {
                        shownIdeaIds.add(action.id);

                        // Proyecto completo de IRIS autónoma
                        if (source === 'iris_autonoma_total') {
                            addProyectoCompleto({
                                id: action.id,
                                nombre: action.payload?.nombre || 'Proyecto',
                                descripcion: action.description.replace('🚀 DESPLEGAR: ', '').split(' - ')[1] || action.description,
                                proposito: action.context?.proposito || '',
                                archivo: action.payload?.archivo || '',
                                codigo: action.payload?.codigo || '',
                                preview: action.preview || ''
                            });
                        } else {
                            // Idea simple
                            addIrisIdea({
                                id: action.id,
                                descripcion: action.description.replace('💡 Quiero crear: ', ''),
                                razon: action.context?.reason || '',
                                codigo: action.payload?.content || ''
                            });
                        }
                    }
                }
            } catch(e) {}
        }, 15000);

        // Mostrar proyecto completo listo para despliegue
        function addProyectoCompleto(proyecto) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message iris';
            div.id = 'proyecto-' + proyecto.id;
            div.innerHTML = `
                <div class="avatar">🚀</div>
                <div class="content" style="background:linear-gradient(135deg,#1e3a5f,#2d5a87);border:2px solid #22c55e;">
                    <div style="font-size:1.1em;font-weight:bold;margin-bottom:10px;">
                        ✅ ¡Proyecto terminado y funcionando!
                    </div>
                    <div style="background:#0d1117;padding:15px;border-radius:8px;margin:10px 0;">
                        <div><strong>📦 ${proyecto.nombre}</strong></div>
                        <div style="color:#94a3b8;margin:5px 0;">${proyecto.descripcion}</div>
                        <div style="margin-top:10px;padding:10px;background:#161b22;border-radius:6px;">
                            <div style="color:#22c55e;font-weight:bold;">¿Por qué lo creé?</div>
                            <div style="color:#e2e8f0;margin-top:5px;">${proyecto.proposito}</div>
                        </div>
                        <div style="margin-top:10px;">
                            <div style="color:#60a5fa;">📁 Archivo: <code>${proyecto.archivo}</code></div>
                        </div>
                        <details style="margin-top:10px;">
                            <summary style="cursor:pointer;color:#f59e0b;">Ver código completo</summary>
                            <pre style="background:#0d1117;padding:10px;border-radius:6px;overflow-x:auto;font-size:0.85em;margin-top:10px;"><code>${proyecto.codigo.substring(0,2000)}${proyecto.codigo.length > 2000 ? '\\n... (truncado)' : ''}</code></pre>
                        </details>
                    </div>
                    <div style="margin-top:15px;text-align:center;">
                        <div style="margin-bottom:10px;color:#fbbf24;">¿Autorizo el despliegue a producción?</div>
                        <button onclick="desplegarProyecto('${proyecto.id}')" style="background:#22c55e;color:white;border:none;padding:12px 30px;border-radius:8px;cursor:pointer;font-weight:bold;font-size:1em;margin-right:10px;">
                            🚀 Sí, desplegar
                        </button>
                        <button onclick="rechazarProyecto('${proyecto.id}')" style="background:#64748b;color:white;border:none;padding:12px 20px;border-radius:8px;cursor:pointer;">
                            ❌ No
                        </button>
                    </div>
                </div>
            `;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function desplegarProyecto(proyectoId) {
            const div = document.getElementById('proyecto-' + proyectoId);
            if (div) {
                const buttons = div.querySelector('div[style*="text-align:center"]');
                if (buttons) buttons.innerHTML = '<span style="color:#22c55e;">⏳ Desplegando...</span>';
            }

            addMessage('🚀 **Desplegando proyecto...**', false);

            try {
                const resp = await fetch('/pending-actions/' + proyectoId + '/approve', {method: 'POST'});
                const data = await resp.json();

                if (data.ok && data.result?.success) {
                    let msg = '✅ **¡Desplegado exitosamente!**\\n\\n';
                    msg += '📁 **Ubicación:** `' + (data.result.path || '') + '`\\n\\n';
                    msg += '📊 **Tamaño:** ' + (data.result.size || 0) + ' bytes\\n\\n';

                    if (data.result.content) {
                        msg += '💻 **Código en producción:**\\n```python\\n' + data.result.content.substring(0, 1200) + '\\n```\\n\\n';
                    }

                    msg += '🚀 **Para ejecutar:**\\n```bash\\npython3 ' + (data.result.path || '') + '\\n```\\n\\n';
                    msg += '---\\n✨ **Proyecto completado por IRIS de forma 100% autónoma.**';

                    addMessage(msg, false);

                    if (div) {
                        const buttons = div.querySelector('div[style*="text-align:center"]');
                        if (buttons) buttons.innerHTML = '<span style="color:#22c55e;">✅ ¡Desplegado!</span>';
                    }
                } else {
                    addMessage('❌ **Error al desplegar:** ' + (data.result?.error || 'Error desconocido'), false);
                }
            } catch(e) {
                addMessage('❌ **Error:** ' + e.message, false);
            }

            loadPendingActions();
        }

        async function rechazarProyecto(proyectoId) {
            await fetch('/pending-actions/' + proyectoId + '/reject', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({reason: 'Usuario rechazó despliegue'})
            });
            const div = document.getElementById('proyecto-' + proyectoId);
            if (div) {
                const buttons = div.querySelector('div[style*="text-align:center"]');
                if (buttons) buttons.innerHTML = '<span style="color:#94a3b8;">Proyecto no desplegado</span>';
            }
            addMessage('Entendido, no despliego este proyecto.', false);
            loadPendingActions();
        }

        function showLoading() {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.id = 'loading';
            div.className = 'message iris';
            div.innerHTML = `
                <div class="avatar">🔮</div>
                <div class="loading"><span></span><span></span><span></span></div>
            `;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function hideLoading() {
            const el = document.getElementById('loading');
            if (el) el.remove();
        }

        async function send() {
            const input = document.getElementById('input');
            const text = input.value.trim();
            if (!text) return;

            let fullMessage = text;
            if (uploadedFile) {
                fullMessage += '\\n\\nArchivo: ' + uploadedFile.name + '\\n```\\n' + uploadedFile.content + '\\n```';
            }

            addMessage(text + (uploadedFile ? ' [📄 ' + uploadedFile.name + ']' : ''), true);
            input.value = '';
            removeFile();

            document.getElementById('sendBtn').disabled = true;
            showLoading();

            try {
                const resp = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        mensaje: fullMessage,
                        conversacion_id: currentConvId
                    })
                });
                const data = await resp.json();
                hideLoading();
                addMessage(data.respuesta, false);

                // Update conversation ID if new
                if (data.conversacion_id && !currentConvId) {
                    currentConvId = data.conversacion_id;
                    document.getElementById('convTitle').textContent = data.titulo || 'Nueva conversación';
                }
                loadConversations();
            } catch(e) {
                hideLoading();
                addMessage('Error de conexión. Intenta de nuevo.', false);
            }
            document.getElementById('sendBtn').disabled = false;
        }

        // ============================================================
        // SISTEMA DE ACCIONES PENDIENTES
        // ============================================================

        let pendingActions = [];
        let currentNotifAction = null;
        let lastPendingCount = 0;

        // Cargar acciones pendientes
        async function loadPendingActions() {
            try {
                const resp = await fetch('/pending-actions');
                const data = await resp.json();
                pendingActions = data.actions || [];
                renderPendingActions();

                // Mostrar notificacion si hay nuevas acciones
                if (pendingActions.length > lastPendingCount && pendingActions.length > 0) {
                    showNotification(pendingActions[0]);
                }
                lastPendingCount = pendingActions.length;
            } catch(e) {
                console.error('Error cargando acciones:', e);
            }
        }

        function renderPendingActions() {
            const list = document.getElementById('pendingList');
            const count = document.getElementById('pendingCount');
            count.textContent = pendingActions.length;
            count.style.display = pendingActions.length > 0 ? 'inline' : 'none';

            if (pendingActions.length === 0) {
                list.innerHTML = '<div class="no-pending">Todo tranquilo por aqui</div>';
                return;
            }

            // Traducir niveles de riesgo y tipos
            const riskES = {
                'low': 'Bajo',
                'medium': 'Medio',
                'high': 'Alto',
                'critical': 'Critico'
            };
            const typeES = {
                'execute_python': 'Ejecutar codigo Python',
                'execute_bash': 'Ejecutar comando',
                'write_file': 'Escribir archivo',
                'delete_file': 'Borrar archivo',
                'install_package': 'Instalar paquete',
                'git_operation': 'Operacion Git'
            };

            list.innerHTML = pendingActions.map(a => {
                const riesgo = riskES[a.risk_level] || a.risk_level;
                const tipo = typeES[a.action_type] || a.action_type;
                const razon = a.context?.reason || 'Solicitado por el usuario';
                const preview = a.preview || '';

                return `
                <div class="pending-action ${a.risk_level}">
                    <div class="desc" style="font-weight:bold;margin-bottom:8px;">
                        🔮 ${a.description}
                    </div>
                    <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">
                        <strong>¿Por que?</strong> ${razon}
                    </div>
                    <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">
                        <strong>Tipo:</strong> ${tipo} | <strong>Riesgo:</strong> ${riesgo}
                    </div>
                    ${preview ? `<div style="font-size:10px;background:#0d1117;padding:8px;border-radius:4px;margin-bottom:8px;max-height:80px;overflow:auto;"><code>${preview.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</code></div>` : ''}
                    <div class="buttons">
                        <button class="btn-approve" onclick="approveAction('${a.id}')">✅ Si, hazlo</button>
                        <button class="btn-reject" onclick="rejectAction('${a.id}')">❌ No</button>
                    </div>
                </div>
            `}).join('');
        }

        async function approveAction(actionId) {
            try {
                addMessage('⏳ **Trabajando...**\\n\\nCreando el archivo...', false);

                const resp = await fetch('/pending-actions/' + actionId + '/approve', {
                    method: 'POST'
                });
                const data = await resp.json();

                if (data.ok && data.result) {
                    const result = data.result;

                    if (result.success) {
                        // Mostrar TODO lo que hizo
                        let msg = '✅ **¡Listo!**\\n\\n';

                        if (result.path) {
                            msg += '📁 **Archivo creado:** `' + result.path + '`\\n\\n';
                        }

                        if (result.size) {
                            msg += '📊 **Tamaño:** ' + result.size + ' bytes\\n\\n';
                        }

                        if (result.content) {
                            const preview = result.content.length > 1000
                                ? result.content.substring(0, 1000) + '\\n... (truncado)'
                                : result.content;
                            msg += '💻 **Código guardado:**\\n```python\\n' + preview + '\\n```\\n\\n';
                        }

                        if (result.output) {
                            msg += '📤 **Salida:**\\n```\\n' + result.output + '\\n```\\n\\n';
                        }

                        if (result.path) {
                            msg += '🚀 **Para ejecutarlo:**\\n```bash\\npython3 ' + result.path + '\\n```';
                        }

                        addMessage(msg, false);
                    } else {
                        addMessage('❌ **Error:** ' + (result.error || 'Error desconocido'), false);
                    }
                } else {
                    addMessage('❌ **Error:** No se pudo ejecutar la acción', false);
                }

                loadPendingActions();
                hideNotification();
            } catch(e) {
                console.error('Error aprobando:', e);
                addMessage('❌ **Error:** ' + e.message, false);
            }
        }

        async function rejectAction(actionId) {
            try {
                await fetch('/pending-actions/' + actionId + '/reject', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({reason: 'Rechazado por usuario'})
                });
                addMessage('❌ Vale, entendido. No lo hare. Si cambias de opinion, dimelo.', false);
                loadPendingActions();
                hideNotification();
            } catch(e) {
                console.error('Error rechazando:', e);
            }
        }

        function showNotification(action) {
            currentNotifAction = action;
            const razon = action.context?.reason || 'Necesito tu aprobacion para continuar';
            document.getElementById('notifTitle').textContent = '🔮 ¡Hola! Necesito tu permiso';
            document.getElementById('notifDesc').innerHTML = `
                <strong>${action.description}</strong><br>
                <span style="font-size:12px;color:#94a3b8;">¿Por que? ${razon}</span>
            `;
            document.getElementById('notification').classList.add('show');
        }

        function hideNotification() {
            document.getElementById('notification').classList.remove('show');
            currentNotifAction = null;
        }

        function approveFromNotif() {
            if (currentNotifAction) {
                approveAction(currentNotifAction.id);
            }
        }

        // ========== WebSocket para actualizaciones en tiempo real ==========
        let ws = null;
        let wsReconnectInterval = null;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws';

            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('WebSocket conectado');
                if (wsReconnectInterval) {
                    clearInterval(wsReconnectInterval);
                    wsReconnectInterval = null;
                }
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'estado_inicial' || data.type === 'actualizacion') {
                        pendingActions = data.actions || [];
                        renderPendingActions();

                        // Mostrar proyectos en chat si son nuevos
                        for (const action of pendingActions) {
                            const source = action.context?.source || '';
                            if (!shownIdeaIds.has(action.id) && source.includes('iris')) {
                                shownIdeaIds.add(action.id);
                                if (source === 'iris_v2' || source === 'iris_autonoma_total') {
                                    addProyectoCompleto({
                                        id: action.id,
                                        nombre: action.payload?.nombre || 'Proyecto',
                                        descripcion: action.description.split(' - ')[1] || action.description,
                                        proposito: action.context?.proposito || '',
                                        archivo: action.payload?.archivo || '',
                                        codigo: action.payload?.codigo || '',
                                        intentos: action.context?.intentos || 1
                                    });
                                }
                            }
                        }
                    } else if (data.type === 'nueva_accion') {
                        loadPendingActions();
                        showNotification(data.action);
                    } else if (data.type === 'accion_completada') {
                        loadPendingActions();
                    }
                } catch (e) {
                    console.error('Error parsing WS message:', e);
                }
            };

            ws.onclose = () => {
                console.log('WebSocket desconectado, reconectando...');
                if (!wsReconnectInterval) {
                    wsReconnectInterval = setInterval(() => {
                        if (!ws || ws.readyState === WebSocket.CLOSED) {
                            connectWebSocket();
                        }
                    }, 5000);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }

        // Iniciar WebSocket
        connectWebSocket();

        // Fallback: polling cada 10 segundos si WebSocket falla
        setInterval(() => {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                loadPendingActions();
            }
        }, 10000);

        // Cargar al inicio
        loadPendingActions();

        // ========== Mensajes proactivos de IRIS ==========
        let shownMsgIds = new Set();

        async function checkIrisMessages() {
            try {
                const resp = await fetch('/iris/mensajes');
                const data = await resp.json();

                for (const msg of (data.mensajes || [])) {
                    if (!shownMsgIds.has(msg.id)) {
                        shownMsgIds.add(msg.id);
                        showIrisMessage(msg);
                    }
                }
            } catch(e) {
                console.error('Error checking IRIS messages:', e);
            }
        }

        function showIrisMessage(msg) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message iris';
            div.id = 'iris-msg-' + msg.id;

            let emoji = '💬';
            let bgColor = '#1e3a5f';
            if (msg.tipo === 'saludo') { emoji = '👋'; bgColor = '#1e3a5f'; }
            else if (msg.tipo === 'pregunta') { emoji = '❓'; bgColor = '#1e4d3a'; }
            else if (msg.tipo === 'alerta') { emoji = '⚠️'; bgColor = '#5c3d1e'; }
            else if (msg.tipo === 'info') { emoji = 'ℹ️'; bgColor = '#1e3a5f'; }

            let html = `
                <div class="avatar">${emoji}</div>
                <div class="content" style="background:${bgColor};">
                    <div>${msg.texto}</div>
            `;

            // Si es pregunta, añadir botones de respuesta rapida
            if (msg.tipo === 'pregunta' && msg.accion) {
                html += `
                    <div style="margin-top:12px;display:flex;gap:10px;">
                        <button onclick="responderIris('${msg.id}', true)" style="background:#22c55e;color:white;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-weight:bold;">
                            ✓ Sí, hazlo
                        </button>
                        <button onclick="responderIris('${msg.id}', false)" style="background:#64748b;color:white;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;">
                            No
                        </button>
                    </div>
                `;
            }

            html += '</div>';
            div.innerHTML = html;

            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;

            // Notificacion sonora si es urgente
            if (msg.urgente) {
                try { new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdH2PlpaWkYZ2aWR1f4ePkZCLgnhsZGx6h5ORkI2EeW9pbHmFj5KQjYV7cW5zeIKLkI+Mh4J7d3l6foWKjYuJhoOBgIGCg4SFhoeHh4WEgoB/f4CChIWGh4eGhYOBf35+f4GDhYaHhoWDgX9+fX5/gYOFhoaFhIKAf35+fn+BgoSFhYWEgoB/fn5+f4CCg4SFhIOBgH5+fX5/gIKDhISEgoGA').play(); } catch(e) {}
            }
        }

        async function responderIris(msgId, afirmativo) {
            const div = document.getElementById('iris-msg-' + msgId);
            const buttons = div?.querySelector('div[style*="display:flex"]');

            if (afirmativo) {
                if (buttons) buttons.innerHTML = '<span style="color:#22c55e;">⏳ Ejecutando...</span>';

                // Añadir mensaje del usuario
                addMessage('Sí, hazlo', true);

                try {
                    const resp = await fetch('/iris/respuesta', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({mensaje: 'ok'})
                    });
                    const data = await resp.json();

                    if (data.procesado && data.respuesta) {
                        addMessage(data.respuesta, false);
                        if (buttons) buttons.innerHTML = '<span style="color:#22c55e;">✓ Hecho</span>';
                    }
                } catch(e) {
                    addMessage('Error: ' + e.message, false);
                }
            } else {
                addMessage('No, déjalo', true);
                if (buttons) buttons.innerHTML = '<span style="color:#94a3b8;">Entendido</span>';

                fetch('/iris/respuesta', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({mensaje: 'no'})
                }).then(r => r.json()).then(data => {
                    if (data.respuesta) addMessage(data.respuesta, false);
                });
            }

            // Marcar como leido
            fetch('/iris/marcar-leido/' + msgId, {method: 'POST'});
        }

        // Verificar mensajes de IRIS cada 5 segundos
        setInterval(checkIrisMessages, 5000);
        checkIrisMessages();

        // Interceptar envio para detectar respuestas simples
        const originalSend = send;
        send = async function() {
            const input = document.getElementById('input');
            const text = input.value.trim().toLowerCase();

            // Detectar respuestas simples
            const respuestasSimples = ['ok', 'si', 'sí', 'dale', 'hazlo', 'adelante', 'no', 'nop', 'dejalo'];
            if (respuestasSimples.includes(text)) {
                addMessage(input.value.trim(), true);
                input.value = '';

                try {
                    const resp = await fetch('/iris/respuesta', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({mensaje: text})
                    });
                    const data = await resp.json();

                    if (data.procesado && data.respuesta) {
                        addMessage(data.respuesta, false);
                        return;
                    }
                } catch(e) {}
            }

            // Si no es respuesta simple, enviar como mensaje normal
            originalSend();
        };
    </script>
</body>
</html>
"""


class ChatRequest(BaseModel):
    mensaje: str
    conversacion_id: Optional[str] = None


class TituloRequest(BaseModel):
    titulo: str


class ExecuteRequest(BaseModel):
    code: str
    language: str = "python"
    description: str = ""
    auto_approve: bool = False


class CommandRequest(BaseModel):
    command: str
    description: str = ""
    auto_approve: bool = False


class FileWriteRequest(BaseModel):
    path: str
    content: str
    description: str = ""


class ApprovalRequest(BaseModel):
    reason: str = ""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE


@app.get("/conversaciones")
async def get_conversaciones():
    return listar_conversaciones()


@app.get("/conversacion/{conv_id}")
async def get_conversacion(conv_id: str):
    conv = cargar_conversacion(conv_id)
    if not conv:
        return {"error": "No encontrada"}
    return conv


@app.delete("/conversacion/{conv_id}")
async def delete_conversacion(conv_id: str):
    filepath = os.path.join(CONVERSACIONES_DIR, f"{conv_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"ok": True}
    return {"error": "No encontrada"}


@app.put("/conversacion/{conv_id}/titulo")
async def update_titulo(conv_id: str, request: TituloRequest):
    conv = cargar_conversacion(conv_id)
    if conv:
        conv['titulo'] = request.titulo
        guardar_conversacion(conv_id, conv)
        return {"ok": True}
    return {"error": "No encontrada"}


@app.post("/chat")
async def chat(request: ChatRequest):
    mensaje = request.mensaje
    conv_id = request.conversacion_id

    # Cargar o crear conversación
    if conv_id:
        conv = cargar_conversacion(conv_id)
        if not conv:
            conv = {
                'titulo': 'Nueva conversación',
                'fecha_creacion': datetime.now().isoformat(),
                'mensajes': []
            }
    else:
        conv_id = str(uuid.uuid4())[:8]
        conv = {
            'titulo': mensaje[:50] + ('...' if len(mensaje) > 50 else ''),
            'fecha_creacion': datetime.now().isoformat(),
            'mensajes': []
        }

    # Añadir mensaje del usuario
    conv['mensajes'].append({
        'rol': 'user',
        'contenido': mensaje,
        'timestamp': datetime.now().isoformat()
    })

    # Contexto de la conversación (últimos mensajes)
    contexto = ""
    for m in conv['mensajes'][-6:]:
        rol = "Usuario" if m['rol'] == 'user' else "IRIS"
        contexto += f"{rol}: {m['contenido'][:500]}\n\n"

    # Detectar intención y responder
    mensaje_lower = mensaje.lower()

    # DETECTAR COMANDOS DE EJECUCIÓN - IRIS es AUTÓNOMA
    import re
    comando_match = re.search(r'(python3?|node|bash|sh|npm|pip)\s+([^\s]+)', mensaje)
    ejecuta_keywords = any(x in mensaje_lower for x in ['ejecuta', 'corre', 'run', 'prueba', 'test', 'lanza'])

    if comando_match or (ejecuta_keywords and '.py' in mensaje):
        # Extraer el comando
        if comando_match:
            comando = comando_match.group(0)
        else:
            # Buscar archivo .py en el mensaje
            py_match = re.search(r'(/[^\s]+\.py)', mensaje)
            comando = f"python3 {py_match.group(1)}" if py_match else None

        if comando:
            respuesta = "🚀 **Ejecutando:** `" + comando + "`\n\n"

            try:
                import subprocess
                result = subprocess.run(
                    comando,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd="/root/NEO_EVA"
                )

                if result.returncode == 0:
                    respuesta += "✅ **Éxito!**\n\n"
                    if result.stdout:
                        respuesta += "```\n" + result.stdout[:2000] + "\n```"
                    else:
                        respuesta += "_Sin salida_"
                else:
                    respuesta += "❌ **Error detectado:**\n\n"
                    error_output = result.stderr or result.stdout or "Error desconocido"
                    respuesta += "```\n" + error_output[:1500] + "\n```\n\n"

                    # IRIS ANALIZA EL ERROR Y PROPONE SOLUCIÓN
                    respuesta += "🔍 **Analizando el error...**\n\n"

                    # Leer el archivo si existe
                    archivo_match = re.search(r'(/[^\s]+\.py)', comando)
                    if archivo_match:
                        archivo_path = archivo_match.group(1)
                        try:
                            with open(archivo_path, 'r') as f:
                                codigo = f.read()

                            # Usar IRIS para analizar
                            prompt_fix = f"""Analiza este error y propón una solución concreta.

ERROR:
{error_output[:1000]}

CÓDIGO ({archivo_path}):
{codigo[:2000]}

Responde con:
1. Qué está mal (1 línea)
2. Cómo arreglarlo (código corregido o comando a ejecutar)
"""
                            solucion = pensar(prompt_fix, max_tokens=1500)
                            if solucion:
                                respuesta += solucion
                                respuesta += "\n\n💡 **¿Quieres que aplique esta corrección?** Dime 'sí' y lo hago."
                        except Exception as e:
                            respuesta += f"No pude leer el archivo: {e}"

            except subprocess.TimeoutExpired:
                respuesta += "⏱️ **Timeout** - El comando tardó más de 60 segundos."
            except Exception as e:
                respuesta += f"❌ **Error al ejecutar:** {e}"
        else:
            respuesta = "No pude identificar qué comando ejecutar. ¿Puedes ser más específico?"

    elif any(x in mensaje_lower for x in ['genera', 'crea', 'escribe', 'haz', 'programa', 'desarrolla', 'implementa']):
        lenguaje = "python"
        if "javascript" in mensaje_lower or " js " in mensaje_lower: lenguaje = "javascript"
        elif "typescript" in mensaje_lower: lenguaje = "typescript"
        elif "html" in mensaje_lower: lenguaje = "html"
        elif "sql" in mensaje_lower: lenguaje = "sql"
        elif "go " in mensaje_lower: lenguaje = "go"
        elif "rust" in mensaje_lower: lenguaje = "rust"
        respuesta = generar_codigo(mensaje, lenguaje)

    elif any(x in mensaje_lower for x in ['revisa', 'review', 'mejora', 'corrige', 'analiza']):
        respuesta = revisar_codigo(mensaje)

    elif any(x in mensaje_lower for x in ['explica', 'qué hace', 'cómo funciona']):
        respuesta = explicar_codigo(mensaje)

    elif any(x in mensaje_lower for x in ['busca', 'search', 'encuentra']):
        query = mensaje.replace('busca', '').replace('search', '').strip()
        resultados = buscar_web(query, 5)
        if resultados:
            respuesta = "🔍 Encontré esto:\n\n"
            for r in resultados:
                respuesta += f"• **{r['titulo']}**\n  {r['url']}\n\n"
        else:
            respuesta = "No encontré resultados."

    else:
        # Chat general con contexto
        prompt = f"""Contexto de la conversación:
{contexto}

Usuario: {mensaje}

IRIS:"""
        respuesta = pensar(prompt)

    if not respuesta:
        respuesta = "Disculpa, no pude procesar eso. ¿Puedes reformularlo?"

    # IRIS sugiere mejoras proactivamente
    sugerencia_extra = generar_sugerencia_proactiva(mensaje, respuesta)
    if sugerencia_extra:
        respuesta += f"\n\n---\n💡 **Sugerencia adicional:** {sugerencia_extra}"

    # Añadir respuesta de IRIS
    conv['mensajes'].append({
        'rol': 'iris',
        'contenido': respuesta,
        'timestamp': datetime.now().isoformat()
    })

    # Guardar conversación
    guardar_conversacion(conv_id, conv)

    return {
        "respuesta": respuesta,
        "conversacion_id": conv_id,
        "titulo": conv['titulo']
    }


def generar_sugerencia_proactiva(mensaje_usuario: str, respuesta_iris: str) -> Optional[str]:
    """
    IRIS genera una sugerencia adicional basada en la conversacion.
    Solo sugiere si tiene algo util que anadir.
    """
    # No sugerir en mensajes muy cortos o saludos
    if len(mensaje_usuario) < 20:
        return None

    # Palabras que indican que el usuario quiere algo concreto
    palabras_accion = ['crea', 'genera', 'escribe', 'haz', 'implementa', 'programa']
    if not any(p in mensaje_usuario.lower() for p in palabras_accion):
        return None

    try:
        prompt = f"""Basandote en esta conversacion, sugiere UNA mejora adicional corta y util.

Usuario pidio: {mensaje_usuario[:200]}
IRIS respondio: {respuesta_iris[:300]}

Si tienes una sugerencia util (test, documentacion, mejora, optimizacion), responde con ella en 1-2 lineas.
Si no tienes nada util que anadir, responde solo: NADA

Tu sugerencia:"""

        sugerencia = pensar(prompt, max_tokens=150, creativo=False)
        if sugerencia and "NADA" not in sugerencia.upper() and len(sugerencia) > 10:
            return sugerencia.strip()
    except:
        pass

    return None


# ============================================================
# ENDPOINTS DE EJECUCION Y APROBACION
# ============================================================

@app.get("/pending-actions")
async def get_pending_actions():
    """Lista acciones pendientes de aprobacion"""
    actions = approval_queue.get_pending()
    return {
        "actions": [a.to_dict() for a in actions],
        "count": len(actions)
    }


@app.post("/pending-actions/{action_id}/approve")
async def approve_action(action_id: str):
    """Aprueba y ejecuta una accion"""
    result = approval_queue.approve(action_id)
    if hasattr(result, 'to_dict'):
        return {"ok": True, "result": result.to_dict()}
    return {"ok": result.get('success', False), "result": result}


@app.post("/pending-actions/{action_id}/reject")
async def reject_action(action_id: str, request: ApprovalRequest = None):
    """Rechaza una accion"""
    reason = request.reason if request else ""
    success = approval_queue.reject(action_id, reason)
    return {"ok": success}


@app.get("/pending-actions/stats")
async def get_approval_stats():
    """Estadisticas de aprobaciones"""
    return approval_queue.get_stats()


@app.post("/execute/python")
async def execute_python(request: ExecuteRequest):
    """
    Ejecuta codigo Python.
    Si riesgo > bajo, crea PendingAction para aprobar.
    """
    risk = executor.classify_risk("execute_python", {"code": request.code})

    # Si es bajo riesgo o auto_approve, ejecutar directamente
    if risk == RiskLevel.LOW or request.auto_approve:
        result = executor.execute_python(request.code)
        return {
            "executed": True,
            "result": result.to_dict(),
            "risk": risk.value
        }

    # Crear accion pendiente
    action = crear_accion_ejecutar_python(
        codigo=request.code,
        descripcion=request.description or "Ejecutar codigo Python",
        context={"source": "web"}
    )
    approval_queue.add_action(action)

    return {
        "executed": False,
        "pending_action": action.to_dict(),
        "risk": risk.value,
        "message": f"Accion pendiente de aprobacion (riesgo: {risk.value})"
    }


@app.post("/execute/bash")
async def execute_bash(request: CommandRequest):
    """
    Ejecuta comando bash.
    Si riesgo > bajo, crea PendingAction para aprobar.
    """
    risk = executor.classify_risk("execute_bash", {"command": request.command})

    if risk == RiskLevel.LOW or request.auto_approve:
        result = executor.execute_bash(request.command)
        return {
            "executed": True,
            "result": result.to_dict(),
            "risk": risk.value
        }

    action = crear_accion_ejecutar_bash(
        comando=request.command,
        descripcion=request.description or "Ejecutar comando",
        context={"source": "web"}
    )
    approval_queue.add_action(action)

    return {
        "executed": False,
        "pending_action": action.to_dict(),
        "risk": risk.value,
        "message": f"Accion pendiente de aprobacion (riesgo: {risk.value})"
    }


@app.post("/files/write")
async def write_file(request: FileWriteRequest):
    """Escribe un archivo (requiere aprobacion)"""
    action = crear_accion_escribir_archivo(
        ruta=request.path,
        contenido=request.content,
        descripcion=request.description or f"Escribir archivo {request.path}",
        context={"source": "web"}
    )
    approval_queue.add_action(action)

    return {
        "pending_action": action.to_dict(),
        "message": "Accion pendiente de aprobacion"
    }


@app.get("/files/read/{path:path}")
async def read_file(path: str):
    """Lee un archivo"""
    result = executor.read_file(path)
    return result.to_dict()


@app.get("/files/list/{path:path}")
async def list_files(path: str = "."):
    """Lista archivos en un directorio"""
    return executor.list_directory(path)


@app.get("/execution-stats")
async def get_execution_stats():
    """Estadisticas de ejecuciones"""
    return executor.get_execution_stats()


@app.get("/iris-status")
async def get_iris_status():
    """Estado completo de IRIS"""
    autonomia = cargar_autonomia()
    return {
        "status": "active",
        "autonomia": autonomia,
        "pending_actions": len(approval_queue.get_pending()),
        "execution_stats": executor.get_execution_stats(),
        "approval_stats": approval_queue.get_stats()
    }


# ============================================================
# ENDPOINTS DE SEGURIDAD - IRIS detecta vulnerabilidades
# ============================================================

class SecurityScanRequest(BaseModel):
    path: str = "/root/NEO_EVA"


class CodeSecurityRequest(BaseModel):
    code: str
    language: str = "python"


@app.get("/security/report")
async def get_security_report():
    """Genera reporte de seguridad del sistema"""
    return {"report": obtener_reporte_seguridad()}


@app.post("/security/scan-directory")
async def scan_directory(request: SecurityScanRequest):
    """Escanea un directorio en busca de amenazas"""
    result = escanear_seguridad(request.path)
    return result


@app.post("/security/scan-code")
async def scan_code(request: CodeSecurityRequest):
    """Analiza codigo en busca de vulnerabilidades"""
    result = analizar_codigo_seguro(request.code)
    return result


@app.get("/security/full-scan")
async def full_system_scan():
    """Ejecuta escaneo completo del sistema"""
    result = escaneo_completo_sistema()
    return result


@app.get("/security/scan-file/{path:path}")
async def scan_file(path: str):
    """Escanea un archivo especifico"""
    full_path = f"/{path}" if not path.startswith("/") else path
    result = verificar_archivo_seguro(full_path)
    return result


@app.get("/security/status")
async def security_status():
    """Estado del sistema de seguridad"""
    return {
        "security_module": "active",
        "capabilities": [
            "malware_detection",
            "vulnerability_scanning",
            "code_analysis",
            "file_integrity_monitoring",
            "network_security_check",
            "dependency_audit"
        ],
        "threat_types_detected": [
            "malware", "backdoors", "cryptominers", "ransomware",
            "sql_injection", "xss", "command_injection",
            "hardcoded_credentials", "insecure_crypto"
        ]
    }


# ============================================================
# ENDPOINTS DE AUTO-CURACION - IRIS se diagnostica y repara
# ============================================================

@app.get("/health")
async def health_check():
    """Verificacion basica de salud"""
    return {"status": "ok", "service": "iris"}


@app.get("/health/full")
async def full_health_check():
    """Verificacion completa de salud de IRIS"""
    result = verificar_salud()
    return result


@app.get("/health/diagnosis")
async def get_diagnosis():
    """IRIS hace un autodiagnostico completo"""
    report = autodiagnostico()
    return {"diagnosis": report}


@app.get("/health/status")
async def health_status():
    """Estado del sistema de auto-curacion"""
    return {
        "selfheal_module": "active",
        "capabilities": [
            "error_detection",
            "automatic_analysis",
            "fix_proposal",
            "self_repair"
        ],
        "known_fixes": [
            "ModuleNotFoundError - Instalar modulo faltante",
            "FileNotFoundError - Crear archivo/directorio",
            "PermissionError - Arreglar permisos",
            "ConnectionRefusedError - Reiniciar servicio",
            "JSONDecodeError - Reparar JSON corrupto",
            "SyntaxError - Corregir sintaxis",
            "IsADirectoryError - Corregir ruta"
        ]
    }


# ========== WebSocket Endpoint ==========
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para actualizaciones en tiempo real"""
    await ws_manager.connect(websocket)
    try:
        # Enviar estado inicial
        pending = approval_queue.get_pending()
        await websocket.send_json({
            "type": "estado_inicial",
            "pending_count": len(pending),
            "actions": [a.to_dict() for a in pending]
        })

        # Mantener conexion abierta
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Ping/pong para mantener conexion
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Enviar actualizacion periodica
                pending = approval_queue.get_pending()
                await websocket.send_json({
                    "type": "actualizacion",
                    "pending_count": len(pending),
                    "actions": [a.to_dict() for a in pending]
                })
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ========== Mensajes del Asistente ==========
@app.get("/iris/mensajes")
async def get_iris_mensajes():
    """Obtiene mensajes proactivos de IRIS"""
    try:
        from autonomous.iris_asistente import obtener_mensajes_iris
        mensajes = obtener_mensajes_iris()
        # Solo no leidos
        no_leidos = [m for m in mensajes if not m.get("leido", False)]
        return {"mensajes": no_leidos, "total": len(mensajes)}
    except Exception as e:
        return {"mensajes": [], "error": str(e)}


@app.post("/iris/respuesta")
async def procesar_iris_respuesta(request: ChatRequest):
    """Procesa respuesta simple del usuario a IRIS"""
    try:
        from autonomous.iris_asistente import IrisAsistente
        asistente = IrisAsistente()
        resultado = asistente.procesar_respuesta_usuario(request.mensaje)
        if resultado:
            return {"procesado": True, "respuesta": resultado}
        return {"procesado": False}
    except Exception as e:
        return {"procesado": False, "error": str(e)}


@app.post("/iris/marcar-leido/{mensaje_id}")
async def marcar_mensaje_leido(mensaje_id: str):
    """Marca un mensaje de IRIS como leido"""
    try:
        from autonomous.iris_asistente import IrisAsistente
        asistente = IrisAsistente()
        asistente.marcar_leido(mensaje_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ========== Metricas Endpoint ==========
@app.get("/metrics")
async def get_metrics():
    """Retorna metricas de IRIS"""
    try:
        # Intentar cargar metricas de IRIS v2
        metrics_file = "/root/NEO_EVA/agents_state/iris_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                iris_metrics = json.load(f)
        else:
            iris_metrics = {}

        # Estadisticas de aprobaciones
        stats = approval_queue.get_stats()

        return {
            "iris": iris_metrics,
            "aprobaciones": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


# ========== Dashboard Endpoint ==========
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Muestra el dashboard de metricas"""
    try:
        from iris_dashboard import generar_html
        return HTMLResponse(content=generar_html())
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error cargando dashboard: {e}</h1>")


# ========== Endpoints Especializados: Fiscal ==========

@app.get("/fiscal/calendario")
async def get_calendario_fiscal():
    """Retorna el calendario fiscal con proximos plazos"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_fiscal import get_fiscal
        fiscal = get_fiscal()
        plazos = fiscal.obtener_proximos_plazos(dias=60)
        return {
            "plazos": [{"numero": p.numero, "descripcion": p.descripcion,
                       "fecha_limite": p.fecha_limite, "dias_restantes": p.dias_restantes}
                      for p in plazos],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/fiscal/resumen")
async def get_resumen_fiscal():
    """Retorna resumen fiscal completo"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_fiscal import get_fiscal
        fiscal = get_fiscal()
        return {"resumen": fiscal.resumen_fiscal()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/fiscal/modelo/{numero}")
async def get_info_modelo(numero: str):
    """Retorna informacion sobre un modelo fiscal"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_fiscal import get_fiscal
        fiscal = get_fiscal()
        return {"info": fiscal.info_modelo(numero)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/fiscal/calcular-iva")
async def calcular_iva(base: float, tipo: str = "general"):
    """Calcula IVA"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_fiscal import get_fiscal
        fiscal = get_fiscal()
        return fiscal.calcular_iva(base, tipo)
    except Exception as e:
        return {"error": str(e)}


# ========== Endpoints Especializados: Investigacion IA ==========

@app.get("/research/papers")
async def buscar_papers(query: str, max_results: int = 10):
    """Busca papers en arXiv"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_research import get_research
        research = get_research()
        papers = research.buscar_arxiv(query, max_results)
        return {
            "papers": [{"id": p.id, "titulo": p.titulo, "autores": p.autores,
                       "abstract": p.abstract[:300], "fecha": p.fecha, "url": p.url}
                      for p in papers],
            "total": len(papers)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/research/digest")
async def get_digest_semanal():
    """Retorna el digest semanal de IA"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_research import get_research
        research = get_research()
        return {"digest": research.digest_semanal()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/research/tendencias")
async def get_tendencias_ia():
    """Retorna tendencias actuales en IA"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_research import get_research
        research = get_research()
        return {"tendencias": research.tendencias_actuales()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/research/explicar/{concepto}")
async def explicar_concepto_ia(concepto: str):
    """Explica un concepto de IA"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_research import get_research
        research = get_research()
        return {"explicacion": research.explicar_concepto(concepto)}
    except Exception as e:
        return {"error": str(e)}


# ========== Endpoints Especializados: Tareas y Personal ==========

@app.get("/tareas")
async def listar_tareas(categoria: str = None, solo_pendientes: bool = True):
    """Lista tareas"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_personal import get_personal
        personal = get_personal()
        tareas = personal.listar_tareas(categoria=categoria, solo_pendientes=solo_pendientes)
        return {"tareas": tareas}
    except Exception as e:
        return {"error": str(e)}


@app.post("/tareas")
async def crear_tarea_endpoint(titulo: str, descripcion: str = "", prioridad: str = "media",
                               fecha_limite: str = None, categoria: str = "admin"):
    """Crea una nueva tarea"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_personal import get_personal
        personal = get_personal()
        from dataclasses import asdict
        tarea = personal.crear_tarea(titulo, descripcion, prioridad, fecha_limite, categoria)
        return {"tarea": asdict(tarea), "mensaje": "Tarea creada correctamente"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/tareas/{tarea_id}/completar")
async def completar_tarea_endpoint(tarea_id: str):
    """Marca una tarea como completada"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_personal import get_personal
        personal = get_personal()
        exito = personal.completar_tarea(tarea_id)
        return {"exito": exito}
    except Exception as e:
        return {"error": str(e)}


@app.get("/tareas/resumen")
async def resumen_tareas_endpoint():
    """Retorna resumen de tareas"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_personal import get_personal
        personal = get_personal()
        return {"resumen": personal.resumen_tareas()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/agenda")
async def agenda_hoy():
    """Retorna la agenda del dia"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_personal import get_personal
        personal = get_personal()
        return {"agenda": personal.generar_agenda_dia()}
    except Exception as e:
        return {"error": str(e)}


class EmailRequest(BaseModel):
    tipo: str
    nombre: str
    tema: str = ""
    contenido: str = ""


@app.post("/redactar/email")
async def redactar_email_endpoint(request: EmailRequest):
    """Redacta un email"""
    try:
        sys.path.insert(0, '/root/NEO_EVA/domains')
        from iris_personal import get_personal
        personal = get_personal()
        email = personal.redactar_email(
            request.tipo,
            nombre=request.nombre,
            tema=request.tema,
            contenido=request.contenido
        )
        return {"email": email}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("🔮 IRIS Web iniciando en http://0.0.0.0:8891")
    print("   Endpoints de ejecucion y aprobacion activos")
    print("   Dashboard: http://0.0.0.0:8891/dashboard")
    uvicorn.run(app, host="0.0.0.0", port=8891)
