function checkURL() {
  const url = document.getElementById('urlInput').value;
  const resultDiv = document.getElementById('result');

  resultDiv.innerHTML = `<span class="text-gray-600">Memeriksa...</span>`;

  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ url })
  })
  .then(res => res.json())
  .then(data => {
    const color = data.label === "malicious" ? "text-red-600" : "text-green-600";
    const labelText = data.label === "malicious" ? "❌ Tautan Berbahaya!" : "✅ Tautan Aman";
    resultDiv.innerHTML = `<p class="${color}">${labelText} <br/>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>`;
  })
  .catch(err => {
    resultDiv.innerHTML = `<span class="text-red-500">Terjadi kesalahan saat memeriksa URL.</span>`;
    console.error(err);
  });
}
