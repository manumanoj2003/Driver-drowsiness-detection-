document.addEventListener('DOMContentLoaded', function () {
    console.log("Drowsiness detection interface is loaded.");

    // Toggle visibility of video feed
    const toggleButton = document.getElementById('toggle-feed');
    const videoFeed = document.getElementById('video-feed');

    toggleButton.addEventListener('click', () => {
        if (videoFeed.style.display === 'none') {
            videoFeed.style.display = 'block';
            toggleButton.textContent = 'Hide Video Feed';
        } else {
            videoFeed.style.display = 'none';
            toggleButton.textContent = 'Show Video Feed';
        }
    });

    // Handle termination of the application
    const terminateButton = document.getElementById('terminate-btn');
    terminateButton.addEventListener('click', () => {
        if (confirm("Are you sure you want to terminate the application?")) {
            fetch('/terminate', { method: 'POST' })
                .then(response => {
                    if (response.ok) {
                        alert("Application terminated.");
                        window.location.reload();
                    } else {
                        alert("Failed to terminate the application.");
                    }
                })
                .catch(error => console.error("Error during termination:", error));
        }
    });
});
