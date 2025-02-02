const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const predictionElement = document.getElementById("prediction");
const confidenceElement = document.getElementById("confidence");
const errorElement = document.getElementById("error");

let ws = null;
let streaming = false;

// Inicializar la cámara
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
    } catch (error) {
        errorElement.textContent = "Error accessing the camera.";
        console.error("Error accessing the camera:", error);
    }
}

// Capturar el frame del video y convertirlo en base64
function captureFrame() {
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg");
}

// Conectar al WebSocket
function connectWebSocket() {
    if (ws) {
        console.log("WebSocket already connected");
        return;
    }

    ws = new WebSocket("ws://127.0.0.1:8000/ws");

    ws.onopen = () => {
        console.log("Connected to WebSocket");
        errorElement.textContent = ""; // Limpiar errores
        streaming = true;
        startStreaming();
    };

    ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        if (response.error) {
            errorElement.textContent = `Server error: ${response.error}`;
        } else {
            errorElement.textContent = ""; // Limpiar errores
            predictionElement.textContent = response.class;
            confidenceElement.textContent = response.confidence.toFixed(2);
        }
    };

    ws.onclose = () => {
        console.log("WebSocket closed");
        errorElement.textContent = "WebSocket connection closed.";
        streaming = false;
        ws = null;
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        errorElement.textContent = "WebSocket connection error.";
        streaming = false;
    };
}

// Función para enviar frames continuamente
function startStreaming() {
    // if (!streaming) return;

    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const frame = captureFrame();
            const message = { frame };
            ws.send(JSON.stringify(message));
            console.log("Frame sent");
        }
    }, 100); // Envía un frame cada 100 ms (~10 fps)
}

// Inicializar la cámara y conectar al WebSocket al cargar la página
window.onload = () => {
    initCamera();
    connectWebSocket();
};
