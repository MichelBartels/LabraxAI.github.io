function GAN(epoch_callback) {
    const BATCH_SIZE = 10;
    const NUMBER_OF_EPOCHS = 10000;
    const LEARNING_RATE_DISCRIMINATOR = 0.00009;
    const LEARNING_RATE_GENERATOR = 0.00009;
    const noise_dimensions = 100;
    const image_size = 784;
    const generator_neurons_layer_1 = 112;
    const generator_neurons_layer_2 = 224;
    const generator_neurons_layer_3 = 336;
    const generator_neurons_layer_4 = 448;
    const generator_neurons_layer_5 = 560;
    const generator_neurons_layer_6 = 672;
    const generator_neurons_layer_7 = 784;
    const generator_structure = [112, 224, 336, 448, 560, 672, image_size];
    const discriminator_structure = [64, 1];

    console.log("Loading MNIST dataset");

    new MNIST(function (MNIST_DATASET) {
        console.log("MNIST dataset loaded");
        let generator_w = [];
        let generator_b = [];
        let generator_matmul = [];
        let generator_add = [];
        let generator_layer = [];
        for (let layer = 0; layer < generator_structure.length; layer++) {
            if (layer == 0) {
                generator_w.push(new Matrix(new_normally_distributed_array(noise_dimensions * generator_structure[layer], 0, 0.001), noise_dimensions, generator_structure[layer]));
                generator_matmul.push(create_multiply_function(BATCH_SIZE, noise_dimensions, generator_structure[layer]));
            } else {
                generator_w.push(new Matrix(new_normally_distributed_array(generator_structure[layer - 1] * generator_structure[layer], 0, 0.001), generator_structure[layer - 1], generator_structure[layer]));
                generator_matmul.push(create_multiply_function(BATCH_SIZE, generator_structure[layer - 1], generator_structure[layer]));
            };
            generator_b.push(new Vector(generator_structure[layer], 0));
            generator_add.push(create_matrix_vector_add_function(BATCH_SIZE, generator_structure[layer]));
            if (layer == (generator_structure.length - 1)) {
                generator_layer.push(create_tanh_function(BATCH_SIZE * generator_structure[layer]));
            } else {
                generator_layer.push(create_relu_function(BATCH_SIZE * generator_structure[layer]));
            };
        };
        
        let discriminator_w = [];
        let discriminator_b = [];
        let discriminator_matmul = [];
        let discriminator_add = [];
        let discriminator_layer = [];
        for (let layer = 0; layer < discriminator_structure.length; layer++) {
            if (layer == 0) {
                discriminator_w.push(new Matrix(new_normally_distributed_array(image_size * discriminator_structure[layer], 0, 0.01), image_size, discriminator_structure[layer]));
                discriminator_matmul.push(create_multiply_function(BATCH_SIZE, image_size, discriminator_structure[layer]));
            } else {
                discriminator_w.push(new Matrix(new_normally_distributed_array(discriminator_structure[layer - 1] * discriminator_structure[layer], 0, 0.01), discriminator_structure[layer - 1], discriminator_structure[layer]));
                discriminator_matmul.push(create_multiply_function(BATCH_SIZE, discriminator_structure[layer - 1], discriminator_structure[layer]));
            };
            discriminator_b.push(new Vector(discriminator_structure[layer], 0));
            discriminator_add.push(create_matrix_vector_add_function(BATCH_SIZE, discriminator_structure[layer]));
            if (layer == (discriminator_structure.length - 1)) {
                discriminator_layer.push(create_sigmoid_function(BATCH_SIZE * discriminator_structure[layer]));
            } else {
                discriminator_layer.push(create_relu_function(BATCH_SIZE * discriminator_structure[layer]));
            };
        };

        let discriminator_slope = [];
        let discriminator_error = [];
        let discriminator_delta = [];
        let discriminator_w_adjustments = [];
        let discriminator_w_adjustments_learning_rate = [];
        let discriminator_w_ascend_gradient = [];
        let discriminator_b_adjustments = [];
        let discriminator_b_adjustments_learning_rate = [];
        let discriminator_b_ascend_gradient = [];
        for (let layer = 0; layer < discriminator_structure.length; layer++) {
            if (layer == (discriminator_structure.length - 1)) {
                discriminator_slope.push(create_tanh_function(BATCH_SIZE, discriminator_structure[layer], true));;
                discriminator_error.push(create_division_function(BATCH_SIZE, discriminator_structure[layer], 1));
            } else {
                discriminator_slope.push(create_relu_function(BATCH_SIZE, discriminator_structure[layer], true));;
                discriminator_error.push(create_multiply_function(BATCH_SIZE, discriminator_structure[layer + 1], discriminator_structure[layer]));
            };
            discriminator_delta.push(create_multiply_function(BATCH_SIZE, discriminator_structure[layer], "matrix"));
            if (layer == 0) {
                discriminator_w_adjustments.push(create_multiply_function(image_size, BATCH_SIZE, discriminator_structure[layer]));
                discriminator_w_adjustments_learning_rate.push(create_multiply_function(image_size, discriminator_structure[layer], "scalar", LEARNING_RATE_DISCRIMINATOR));
                discriminator_w_ascend_gradient.push(create_matrix_matrix_add_function(image_size, discriminator_structure[layer]));
            } else {
                discriminator_w_adjustments.push(create_multiply_function(discriminator_structure[layer - 1], BATCH_SIZE, discriminator_structure[layer]));
                discriminator_w_adjustments_learning_rate.push(create_multiply_function(discriminator_structure[layer - 1], discriminator_structure[layer], "scalar", LEARNING_RATE_DISCRIMINATOR));
                discriminator_w_ascend_gradient.push(create_matrix_matrix_add_function(discriminator_structure[layer - 1], discriminator_structure[layer]));
            };
            discriminator_b_adjustments.push(create_sum_function(BATCH_SIZE, discriminator_structure[layer]));
            discriminator_b_adjustments_learning_rate.push(create_multiply_function(discriminator_structure[layer], 1, "scalar", LEARNING_RATE_DISCRIMINATOR))
            discriminator_b_ascend_gradient.push(create_matrix_matrix_add_function(discriminator_structure[layer], 1));
        };
        let discriminator_error_fake = create_subtract_from_scalar_function(BATCH_SIZE, discriminator_structure[discriminator_structure.length - 1], 1);

        let generator_slope = [];
        let generator_error = [];
        let generator_delta = [];
        let generator_w_adjustments = [];
        let generator_w_adjustments_learning_rate = [];
        let generator_w_descend_gradient = [];
        let generator_b_adjustments = [];
        let generator_b_adjustments_learning_rate = [];
        let generator_b_descend_gradient = [];
        for (let layer = 0; layer < generator_structure.length; layer++) {
            if (layer == (generator_structure.length - 1)) {
                generator_slope.push(create_tanh_function(BATCH_SIZE, generator_structure[layer], true));;
                generator_error.push(create_multiply_function(BATCH_SIZE, discriminator_structure[discriminator_structure.length - 1], generator_structure[layer]));
            } else {
                generator_slope.push(create_relu_function(BATCH_SIZE, generator_structure[layer], true));;
                generator_error.push(create_multiply_function(BATCH_SIZE, generator_structure[layer + 1], generator_structure[layer]));
            };
            generator_delta.push(create_multiply_function(BATCH_SIZE, generator_structure[layer], "matrix"));
            if (layer == 0) {
                generator_w_adjustments.push(create_multiply_function(noise_dimensions, BATCH_SIZE, generator_structure[layer]));
                generator_w_adjustments_learning_rate.push(create_multiply_function(noise_dimensions, generator_structure[layer], "scalar", LEARNING_RATE_GENERATOR));
                generator_w_descend_gradient.push(create_matrix_matrix_subtract_function(noise_dimensions, generator_structure[layer]));
            } else {
                generator_w_adjustments.push(create_multiply_function(generator_structure[layer - 1], BATCH_SIZE, generator_structure[layer]));
                generator_w_adjustments_learning_rate.push(create_multiply_function(generator_structure[layer - 1], generator_structure[layer], "scalar", LEARNING_RATE_GENERATOR));
                generator_w_descend_gradient.push(create_matrix_matrix_subtract_function(generator_structure[layer - 1], generator_structure[layer]))
            };
            generator_b_adjustments.push(create_sum_function(BATCH_SIZE, generator_structure[layer]));
            generator_b_adjustments_learning_rate.push(create_multiply_function(generator_structure[layer], 1, "scalar", LEARNING_RATE_GENERATOR))
            generator_b_descend_gradient.push(create_matrix_matrix_subtract_function(generator_structure[layer], 1));
        };

        window.epoch = 0;
        window.interval = setInterval(function() {
            // Classify real images
            let batch = MNIST_DATASET.next_batch(BATCH_SIZE);

            let discriminator_x_no_activation = [];
            let discriminator_x = [];
            for (let layer = 0; layer < (discriminator_structure.length + 1); layer++) {
                if (layer == 0) {
                    discriminator_x.push(batch);
                    discriminator_x_no_activation.push(batch);
                } else {
                    discriminator_x_no_activation.push(discriminator_add[layer - 1](discriminator_matmul[layer - 1](discriminator_x[layer - 1], discriminator_w[layer - 1].array), discriminator_b[layer - 1].array));
                    discriminator_x.push(discriminator_layer[layer - 1](discriminator_x_no_activation[layer]));
                };
            };
            
            // Backpropagate discriminator error of real images
            let discriminator_slope_ = [];
            let discriminator_w_transpose = [];
            let discriminator_layer_transpose = [];
            let discriminator_error_ = [];
            let discriminator_delta_ = [];
            let discriminator_w_adjustments_ = [];
            let discriminator_w_adjustments_learning_rate_ = [];
            let discriminator_b_adjustments_ = [];
            let discriminator_b_adjustments_learning_rate_ = [];
            for (let layer = 0; layer < discriminator_structure.length; layer++) {
                if (layer == discriminator_structure.length - 1) {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x[layer + 1]));
                } else {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x_no_activation[layer + 1]));
                    discriminator_w_transpose.push(discriminator_w[layer + 1].transpose_matrix.array);
                };
                if (layer == 0) {
                    discriminator_layer_transpose.push(new Matrix(discriminator_x[layer], BATCH_SIZE, noise_dimensions).transpose_matrix.array);
                } else {
                    discriminator_layer_transpose.push(new Matrix(discriminator_x[layer], BATCH_SIZE, discriminator_structure[layer]).array);
                };
            };
            
            for (let layer = discriminator_structure.length - 1; layer >= 0; layer--) {
                if (layer == discriminator_structure.length - 1) {
                    discriminator_error_[layer] = discriminator_error[layer](discriminator_x[discriminator_x.length - 1]);
                } else {
                    discriminator_error_[layer] = discriminator_error[layer](discriminator_delta_[layer + 1], discriminator_w_transpose[layer]);
                };
                discriminator_delta_[layer] = discriminator_delta[layer](discriminator_slope_[layer], discriminator_error_[layer]);
                discriminator_w_adjustments_[layer] = discriminator_w_adjustments[layer](discriminator_layer_transpose[layer], discriminator_delta_[layer]);
                discriminator_w_adjustments_learning_rate_[layer] = discriminator_w_adjustments_learning_rate[layer](discriminator_w_adjustments_[layer]);
                discriminator_b_adjustments_[layer] = discriminator_b_adjustments[layer](discriminator_delta_[layer]);
                discriminator_b_adjustments_learning_rate_[layer] = discriminator_b_adjustments_learning_rate[layer](discriminator_b_adjustments_[layer]);
            };

            let discriminator_real_loss = discriminator_error_[discriminator_error_.length - 1];

            // Optimize discriminator for classifying real images
            for (let layer = 0; layer < discriminator_structure.length; layer++) {
                if (layer == 0) {
                    discriminator_w[layer] = new Matrix(discriminator_w_ascend_gradient[layer](discriminator_w[layer].array, discriminator_w_adjustments_learning_rate_[layer]), noise_dimensions, discriminator_structure[layer]); 
                } else {
                    discriminator_w[layer] = new Matrix(discriminator_w_ascend_gradient[layer](discriminator_w[layer].array, discriminator_w_adjustments_learning_rate_[layer]), discriminator_structure[layer - 1], discriminator_structure[layer]);
                };
                discriminator_b[layer] = new Vector(discriminator_b_ascend_gradient[layer](discriminator_b[layer].array, discriminator_b_adjustments_learning_rate_[layer]));
            };
            
            // Generate fake images
            let generator_x_no_activation = [];
            let generator_x = [];
            for (let layer = 0; layer < (generator_structure.length + 1); layer++) {
                if (layer == 0) {
                    generator_x.push(new_random_array(BATCH_SIZE * noise_dimensions, -1, 1));
                    generator_x_no_activation.push(generator_x_no_activation[layer]);
                } else {
                    generator_x_no_activation.push(generator_add[layer - 1](generator_matmul[layer - 1](generator_x[layer - 1], generator_w[layer - 1].array), generator_b[layer - 1].array));
                    generator_x.push(generator_layer[layer - 1](generator_x_no_activation[layer]));
                };
            };

            // Classify fake images
            discriminator_x_no_activation = [];
            discriminator_x = [];
            for (let layer = 0; layer < (discriminator_structure.length + 1); layer++) {
                if (layer == 0) {
                    discriminator_x.push(generator_x[generator_x.length - 1]);
                    discriminator_x_no_activation.push(generator_x[generator_x.length - 1]);
                } else {
                    discriminator_x_no_activation.push(discriminator_add[layer - 1](discriminator_matmul[layer - 1](discriminator_x[layer - 1], discriminator_w[layer - 1].array), discriminator_b[layer - 1].array));
                    discriminator_x.push(discriminator_layer[layer - 1](discriminator_x_no_activation[layer]));
                };
            };

            // Backpropagate discriminator error of fake images
            let discriminator_fake_targets = new Matrix(BATCH_SIZE, 1, -1);
            
            discriminator_slope_ = [];
            discriminator_w_transpose = [];
            discriminator_layer_transpose = [];
            discriminator_error_ = [];
            discriminator_delta_ = [];
            discriminator_w_adjustments_ = [];
            discriminator_w_adjustments_learning_rate_ = [];
            discriminator_b_adjustments_ = [];
            discriminator_b_adjustments_learning_rate_ = [];
            for (let layer = 0; layer < discriminator_structure.length; layer++) {
                if (layer == discriminator_structure.length - 1) {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x[layer + 1]));
                } else {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x_no_activation[layer + 1]));
                    discriminator_w_transpose.push(discriminator_w[layer + 1].transpose_matrix.array);
                };
                if (layer == 0) {
                    discriminator_layer_transpose.push(new Matrix(discriminator_x[layer], BATCH_SIZE, noise_dimensions).transpose_matrix.array);
                } else {
                    discriminator_layer_transpose.push(new Matrix(discriminator_x[layer], BATCH_SIZE, discriminator_structure[layer]).array);
                };
            };
            
            for (let layer = discriminator_structure.length - 1; layer >= 0; layer--) {
                if (layer == discriminator_structure.length - 1) {
                    let discriminator_error_1 = discriminator_error_fake(discriminator_x[discriminator_x.length - 1]);
                    discriminator_error_[layer] = discriminator_error[layer](discriminator_error_1);
                } else {
                    discriminator_error_[layer] = discriminator_error[layer](discriminator_delta_[layer + 1], discriminator_w_transpose[layer]);
                };
                discriminator_delta_[layer] = discriminator_delta[layer](discriminator_slope_[layer], discriminator_error_[layer]);
                discriminator_w_adjustments_[layer] = discriminator_w_adjustments[layer](discriminator_layer_transpose[layer], discriminator_delta_[layer]);
                discriminator_w_adjustments_learning_rate_[layer] = discriminator_w_adjustments_learning_rate[layer](discriminator_w_adjustments_[layer]);
                discriminator_b_adjustments_[layer] = discriminator_b_adjustments[layer](discriminator_delta_[layer]);
                discriminator_b_adjustments_learning_rate_[layer] = discriminator_b_adjustments_learning_rate[layer](discriminator_b_adjustments_[layer]);
            };

            let discriminator_fake_loss = discriminator_error_[discriminator_error_.length - 1];

            // Optimize discriminator for classifying fake images
            for (let layer = 0; layer < discriminator_structure.length; layer++) {
                if (layer == 0) {
                    discriminator_w[layer] = new Matrix(discriminator_w_ascend_gradient[layer](discriminator_w[layer].array, discriminator_w_adjustments_learning_rate_[layer]), noise_dimensions, discriminator_structure[layer]); 
                } else {
                    discriminator_w[layer] = new Matrix(discriminator_w_ascend_gradient[layer](discriminator_w[layer].array, discriminator_w_adjustments_learning_rate_[layer]), discriminator_structure[layer - 1], discriminator_structure[layer]);
                };
                discriminator_b[layer] = new Vector(discriminator_b_ascend_gradient[layer](discriminator_b[layer].array, discriminator_b_adjustments_learning_rate_[layer]));
            };

            // Backprogate generator loss
            let generator_slope_ = [];
            let generator_w_transpose = [];
            let generator_layer_transpose = [];
            let generator_error_ = [];
            let generator_delta_ = [];
            let generator_w_adjustments_ = [];
            let generator_w_adjustments_learning_rate_ = [];
            let generator_b_adjustments_ = [];
            let generator_b_adjustments_learning_rate_ = [];
            for (let layer = 0; layer < generator_structure.length; layer++) {
                if (layer == generator_structure.length - 1) {
                    generator_slope_.push(generator_slope[layer](generator_x[layer + 1]));
                    generator_w_transpose.push(discriminator_w[0].transpose_matrix.array);
                } else {
                    generator_slope_.push(generator_slope[layer](generator_x_no_activation[layer + 1]));
                    generator_w_transpose.push(generator_w[layer + 1].transpose_matrix.array);
                };
                if (layer == 0) {
                    generator_layer_transpose.push(new Matrix(generator_x[layer], BATCH_SIZE, noise_dimensions).transpose_matrix.array);
                } else {
                    generator_layer_transpose.push(new Matrix(generator_x[layer], BATCH_SIZE, generator_structure[layer]).array);
                };
            };
            
            for (let layer = generator_structure.length - 1; layer >= 0; layer--) {
                if (layer == generator_structure.length - 1) {
                    generator_error_[layer] = generator_error[layer](discriminator_delta_[0], discriminator_w[0].array);
                } else {
                    generator_error_[layer] = generator_error[layer](generator_delta_[layer + 1], generator_w_transpose[layer]);
                };
                generator_delta_[layer] = generator_delta[layer](generator_slope_[layer], generator_error_[layer]);
                generator_w_adjustments_[layer] = generator_w_adjustments[layer](generator_layer_transpose[layer], generator_delta_[layer]);
                generator_w_adjustments_learning_rate_[layer] = generator_w_adjustments_learning_rate[layer](generator_w_adjustments_[layer]);
                generator_b_adjustments_[layer] = generator_b_adjustments[layer](generator_delta_[layer]);
                generator_b_adjustments_learning_rate_[layer] = generator_b_adjustments_learning_rate[layer](generator_b_adjustments_[layer]);
            };

            let generator_loss = discriminator_error_[discriminator_error_.length - 1];

            // Optimize generator
            for (let layer = 0; layer < generator_structure.length; layer++) {
                if (layer == 0) {
                    generator_w[layer] = new Matrix(generator_w_descend_gradient[layer](generator_w[layer].array, generator_w_adjustments_learning_rate_[layer]), noise_dimensions, generator_structure[layer]); 
                } else {
                    generator_w[layer] = new Matrix(generator_w_descend_gradient[layer](generator_w[layer].array, generator_w_adjustments_learning_rate_[layer]), generator_structure[layer - 1], generator_structure[layer]);
                };
                generator_b[layer] = new Vector(generator_b_descend_gradient[layer](generator_b[layer].array, generator_b_adjustments_learning_rate_[layer]));
            };

            console.log("Epoch: " + epoch + " Discriminator error real: " + new Matrix(discriminator_real_loss, BATCH_SIZE, 1).mean_squared_error);
            console.log("Epoch: " + epoch + " Discriminator error fake: " + new Matrix(discriminator_fake_loss, BATCH_SIZE, 1).mean_squared_error);
            console.log("Epoch: " + epoch + " Generator error: " + new Matrix(generator_loss, BATCH_SIZE, 1).mean_squared_error);

            console.log(generator_error_);

            epoch_callback(generator_x[generator_x.length - 1], new Matrix(discriminator_real_loss, BATCH_SIZE, 1).mean_squared_error, new Matrix(discriminator_fake_loss, BATCH_SIZE, 1).mean_squared_error, new Matrix(generator_loss, BATCH_SIZE, 1)[0, 0])
            window.epoch++;
        }, 10);
    });
};
function stop_training() {
    clearInterval(window.interval);
};