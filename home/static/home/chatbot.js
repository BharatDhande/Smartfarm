async function sendMessage() {
    const message = document.getElementById('message').value;
    document.getElementById('message').value = '';

    const chatBox = document.getElementById('chat');
    chatBox.innerHTML += `<div><strong>You:</strong> ${message}</div>`;

    try {
        const response = await fetch('/home/chat/', {  // Make sure this URL matches your URL configuration
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();
        chatBox.innerHTML += `<div><strong>AgroVision:</strong> ${data.response}</div>`;
    } catch (error) {
        chatBox.innerHTML += `<div><strong>Error:</strong> Something went wrong. Please try again later.</div>`;
    }
}
