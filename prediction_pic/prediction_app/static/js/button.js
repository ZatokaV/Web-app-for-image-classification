
document.getElementById("upload").addEventListener("change", function() {

    var selectedFile = this.files[0];

    if (selectedFile) {
        document.getElementById("submitBtn").hidden = false;
    } else {
        document.getElementById("submitBtn").hidden = true;
    }
});


document.getElementById("submitBtn").addEventListener("click", function() {

    document.getElementById("submitBtn").hidden = true;
});