// Toggle class active

const navbarNav = document.querySelector(".navbar-nav");
// Ketika humberger menu di klik

document.querySelector("#hamburger-menu").onclick = () => {
    navbarNav.classList.toggle("active");
}

// klik diluar sidebar untuk menghilangkan nav

const hamburger = document.querySelector("#hamburger-menu");

document.addEventListener("click", function (e) {
    if (!hamburger.contains(e.target) && !navbarNav.contains(e.target)) {
        navbarNav.classList.remove("active");
    }
})

// Drop area
document.addEventListener("DOMContentLoaded", function() {
    const dropArea = document.querySelector("#box-drop-area");
    const fileInput = document.querySelector("#file-input");
    const submitButton = document.querySelector("#submit-button");
    const previewContainer = document.querySelector("#preview-container");
    const uploadButton = document.querySelector("#upload-button");
    const instructions = document.querySelector(".instruction");

    // Drag and drop events
    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("dragover");
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dropArea.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        fileInput.files = e.dataTransfer.files;
        handleFile(file);
    });

    // Upload button event
    uploadButton.addEventListener("click", () => {
        fileInput.click();
    });

    // File input change event
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        handleFile(file);
    });

    // Function to handle file
    function handleFile(file) {
        if (file) {
            const reader = new FileReader();
    
            reader.onload = function(event) {
                // Set URL of the uploaded image
                const imageUrl = event.target.result;
            
                // Preview the uploaded image
                previewContainer.style.backgroundImage = `url('${imageUrl}')`;
                previewContainer.style.display = "block";
    
                // Adjust styles after file upload
                dropArea.style.padding = "20px"; // Add padding to drop area
                document.getElementById("box-drop-area").style.display = "flex"; // Ensure box-drop-area is flex
                instructions.style.display = "none"; // Hide instructions
                uploadButton.style.display = "none"; // Hide upload button
                document.getElementById("upload-cloud").style.display = "none"; // Hide upload cloud
                submitButton.style.display = "inline-block"; // Show submit button
    
                // Ensure preview fills the box-drop-area
                previewContainer.style.backgroundSize = "contain";
                previewContainer.style.backgroundRepeat = "no-repeat";
                previewContainer.style.backgroundPosition = "center";
                previewContainer.style.maxWidth = "100%";
                previewContainer.style.maxHeight = "100%";
            };
    
            reader.readAsDataURL(file);
        }
    }
});

