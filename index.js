window.onload = function() {
    document.getElementById("ok").addEventListener("click", function() {
        document.getElementById("settings").style.opacity = 0;
        document.getElementById("settings").style.display = "none";
        document.getElementById("loading").style.display = "block";
        document.getElementById("loading").style.opacity = 1;
        GAN(document.getElementsByTagName("select")[0].value, function(image, discriminator_loss_real, discriminator_loss_fake, generator_loss, learning_rate) {
            showPicture(image, document.getElementById("preview"), 64);
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
            setInterval(function() {
                showPicture(sample(), document.getElementById("preview"), 64);
            }, 1);
        }, 1000);
    });
};