#!/bin/bash
# Detener el Mundo Vivo

if [ -f /tmp/living_world.pid ]; then
    PID=$(cat /tmp/living_world.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Deteniendo Mundo Vivo (PID: $PID)..."
        kill -SIGTERM $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            kill -9 $PID
        fi
        rm /tmp/living_world.pid
        echo "Mundo Vivo detenido."
    else
        echo "Proceso no encontrado."
        rm /tmp/living_world.pid
    fi
else
    # Buscar por nombre
    PID=$(pgrep -f "living_world_daemon.py")
    if [ -n "$PID" ]; then
        echo "Deteniendo Mundo Vivo (PID: $PID)..."
        kill -SIGTERM $PID
        sleep 2
        echo "Mundo Vivo detenido."
    else
        echo "Mundo Vivo no está corriendo."
    fi
fi
