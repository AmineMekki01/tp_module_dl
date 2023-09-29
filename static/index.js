const submitButton = document.getElementById("submit-button");
const fileInput = document.getElementById("inputfile");

const errorMessage = document.getElementById("error-message");

errorMessage.style.textAlign = "center";
errorMessage.style.color = "red";

function isFormValid() {
    const fileValue = fileInput.value;

    return fileValue !== "";
}


submitButton.addEventListener("click", async (event) => {
    event.preventDefault();

    if (!isFormValid()) {
        errorMessage.textContent = "Please upload a file.";
        return;
    }

    if (fileInput.value !== '') {
        await sendRequest();
    }
});


fileInput.addEventListener('change', () => {
    errorMessage.textContent = '';
});

async function sendRequest() {
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch("/image_classifier", {
        method: "POST",
        body: formData  // Here, we send formData instead of JSON
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error:', response.statusText, errorText);
        errorMessage.textContent = 'Server error: ' + response.statusText;
        return;
    }

    // Get the response data
    try {
        const data = await response.json();
        console.log(data);
        const answerElement = document.getElementById("answer");
        answerElement.textContent = data.class_name;
    } catch (error) {
        console.error('JSON parsing error:', error);
    }
}