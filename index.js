window.paused = false;
window.show_samples = true;
window.onload = function() {
    document.getElementById("ok").addEventListener("click", function() {
        document.getElementById("settings").style.opacity = 0;
        document.getElementById("settings").style.display = "none";
        document.getElementById("loading").style.display = "block";
        document.getElementById("loading").style.opacity = 1;
        GAN(document.getElementsByTagName("select")[0].value, function(image, discriminator_loss_real, discriminator_loss_fake, generator_loss, learning_rate, training_images) {
            if (window.show_samples) {
                showPicture(image, document.getElementById("preview"), 64);
            } else {
                showPicture(training_images, document.getElementById("preview"), 64);
            };
            document.getElementById("epoch").innerText = window.epoch;
            document.getElementById("discriminator_loss_real").innerText = discriminator_loss_real;
            document.getElementById("discriminator_loss_fake").innerText = discriminator_loss_fake;
            document.getElementById("generator_loss").innerHTML = generator_loss;
            document.getElementById("learning_rate").innerHTML = learning_rate;
            document.getElementById("train_progress").value = window.epoch;
        }, function() {
            document.getElementById("settings_overlay").style.display = "none";
            document.getElementById("settings_overlay").style.opacity = 0;
        }, function(sample) {
            document.getElementById("training_in_progress").innerText = "Training finished";
            document.getElementById("pause").style.display = "none";
            document.getElementById("switch_canvas").style.display = "none";
        }, 1000);
    });
    document.getElementById("pause").addEventListener("click", function() {
        if (window.paused) {
            window.continue_training();
            document.getElementById("pause").innerHTML = "&#x23f8;";
        } else {
            stop_training();
            document.getElementById("pause").innerHTML = "&#x25b6;";
        };
        window.paused = !window.paused;
    });
    document.getElementById("restart").addEventListener("click", function() {
        location.reload();
    });
    let samples = document.getElementById("samples");
    let training_images = document.getElementById("training_images");
    samples.addEventListener("click", function() {
        samples.className = "selected";
        training_images.className = "not_selected";
        window.show_samples = true;
    });
    training_images.addEventListener("click", function() {
        training_images.className = "selected";
        samples.className = "not_selected";
        window.show_samples = false;
    });
};