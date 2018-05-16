window.onload = function() {
    document.getElementsByTagName("button")[0].addEventListener("click", function() {
        document.getElementById("settings").style["opacity"] = 0;
        document.getElementById("settings").style["display"] = "none";
        document.getElementById("loading").style["display"] = "block";
        document.getElementById("loading").style["opacity"] = 1;
        GAN(document.getElementsByTagName("select")[0].value, function(image, discriminator_loss_real, discriminator_loss_fake, generator_loss, learning_rate) {
            showPicture(image.slice(0, 784 * 64), document.getElementById("preview"), 64);
            document.getElementById("epoch").innerHTML = window.epoch;
            document.getElementById("discriminator_loss_real").innerHTML = discriminator_loss_real;
            document.getElementById("discriminator_loss_fake").innerHTML = discriminator_loss_fake;
            document.getElementById("generator_loss").innerHTML = generator_loss;
            document.getElementById("learning_rate").innerHTML = learning_rate;
        }, function() {
            console.log("called");
            document.getElementById("settings_overlay").style["display"] = "none";
            document.getElementById("settings_overlay").style["opacity"] = 0;
        });
    });
};