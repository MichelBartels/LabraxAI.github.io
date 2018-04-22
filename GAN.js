function GAN(epoch_callback, label) {
    const BATCH_SIZE = 64;
    const NUMBER_OF_EPOCHS = 10000;
    var STEP_SIZE = 0.001;
    const BETA_1 = 0.9;
    const BETA_2 = 0.999;
    const EPSILON = 1e-4;
    const noise_dimensions = 100;
    const image_size = 784;
    const generator_neurons_layer_1 = 112;
    const generator_neurons_layer_2 = 224;
    const generator_neurons_layer_3 = 336;
    const generator_neurons_layer_4 = 448;
    const generator_neurons_layer_5 = 560;
    const generator_neurons_layer_6 = 672;
    const generator_neurons_layer_7 = 784;
    const generator_structure = [128, image_size];
    const discriminator_structure = [128, 1];
    const decay = 0.001;

    console.log("Loading MNIST dataset");

    new MNIST(label, function (MNIST_DATASET) {
        console.log("MNIST dataset loaded");
        window.generator_w = [];
        window.generator_b = [];
        let generator_matmul = [];
        let generator_add = [];
        let generator_layer = [];
        let generator_multiply_learning_rate_w = [];
        let generator_multiply_learning_rate_b = [];
        let generator_subtract_gradients_w = [];
        let generator_subtract_gradients_b = [];
        for (let layer = 0; layer < generator_structure.length; layer++) {
            if (layer == 0) {
                generator_w.push(new Matrix(new_normally_distributed_array(noise_dimensions * generator_structure[layer], 0, Math.sqrt(2 / noise_dimensions)), noise_dimensions, generator_structure[layer]));
                generator_matmul.push(create_multiply_function(BATCH_SIZE, noise_dimensions, generator_structure[layer]));
                generator_multiply_learning_rate_w.push(create_multiply_function(noise_dimensions, generator_structure[layer], "scalar", STEP_SIZE));
                generator_subtract_gradients_w.push(create_matrix_matrix_subtract_function(noise_dimensions, generator_structure[layer]));
            } else {
                if (layer == (generator_structure.length - 1)) {
                    generator_w.push(new Matrix(new_normally_distributed_array(generator_structure[layer - 1] * generator_structure[layer], 0, Math.sqrt(2 / generator_structure[generator_structure.length - 1])), generator_structure[layer - 1], generator_structure[layer]));
                } else {
                    generator_w.push(new Matrix(new_normally_distributed_array(generator_structure[layer - 1] * generator_structure[layer], 0, Math.sqrt(2 / generator_structure[generator_structure.length - 1])), generator_structure[layer - 1], generator_structure[layer]));
                };
                generator_matmul.push(create_multiply_function(BATCH_SIZE, generator_structure[layer - 1], generator_structure[layer]));
                generator_multiply_learning_rate_w.push(create_multiply_function(generator_structure[layer - 1], generator_structure[layer], "scalar", STEP_SIZE));
                generator_subtract_gradients_w.push(create_matrix_matrix_subtract_function(generator_structure[layer - 1], generator_structure[layer]));
            };
            generator_b.push(new Vector(generator_structure[layer], 0));
            generator_add.push(create_matrix_vector_add_function(BATCH_SIZE, generator_structure[layer]));
            if (layer == (generator_structure.length - 1)) {
                generator_layer.push(create_tanh_function(BATCH_SIZE * generator_structure[layer]));
            } else {
                generator_layer.push(create_leaky_relu_function(BATCH_SIZE * generator_structure[layer]));
            };
            generator_subtract_gradients_b.push(create_matrix_matrix_subtract_function(1, generator_structure[layer]));
            generator_multiply_learning_rate_b.push(create_multiply_function(generator_structure[layer], 1, "scalar", STEP_SIZE * -1));
        };
        
        window.discriminator_w = [];
        window.discriminator_b = [];
        let discriminator_matmul = [];
        let discriminator_add = [];
        let discriminator_layer = [];
        let discriminator_subtract_gradients_w = [];
        let discriminator_subtract_gradients_b = [];
        let discriminator_multiply_learning_rate_w = [];
        let discriminator_multiply_learning_rate_b = [];
        for (let layer = 0; layer < discriminator_structure.length; layer++) {
            if (layer == 0) {
                discriminator_w.push(new Matrix(new_normally_distributed_array(image_size * discriminator_structure[layer], 0, Math.sqrt(2 / image_size)), image_size, discriminator_structure[layer]));
                discriminator_matmul.push(create_multiply_function(BATCH_SIZE, image_size, discriminator_structure[layer]));
                discriminator_subtract_gradients_w.push(create_matrix_matrix_subtract_function(image_size, discriminator_structure[layer]));
                discriminator_multiply_learning_rate_w.push(create_multiply_function(image_size, discriminator_structure[layer], "scalar", STEP_SIZE));
            } else {
                if (layer == (generator_structure.length - 1)) {
                    discriminator_w.push(new Matrix(new_normally_distributed_array(discriminator_structure[layer - 1] * discriminator_structure[layer], 0, Math.sqrt(2 / generator_structure[generator_structure.length - 1])), discriminator_structure[layer - 1], discriminator_structure[layer]));
                } else {
                    discriminator_w.push(new Matrix(new_normally_distributed_array(discriminator_structure[layer - 1] * discriminator_structure[layer], 0, Math.sqrt(2 / generator_structure[generator_structure.length - 1])), discriminator_structure[layer - 1], discriminator_structure[layer]));
                }
                discriminator_matmul.push(create_multiply_function(BATCH_SIZE, discriminator_structure[layer - 1], discriminator_structure[layer]));
                discriminator_subtract_gradients_w.push(create_matrix_matrix_subtract_function(discriminator_structure[layer - 1], discriminator_structure[layer]));
                discriminator_multiply_learning_rate_w.push(create_multiply_function(discriminator_structure[layer - 1], discriminator_structure[layer], "scalar", STEP_SIZE));
            };
            discriminator_b.push(new Vector(discriminator_structure[layer], 0));
            discriminator_add.push(create_matrix_vector_add_function(BATCH_SIZE, discriminator_structure[layer]));
            if (layer == (discriminator_structure.length - 1)) {
                discriminator_layer.push(create_sigmoid_function(BATCH_SIZE * discriminator_structure[layer]));
            } else {
                discriminator_layer.push(create_leaky_relu_function(BATCH_SIZE * discriminator_structure[layer]));
            };
            discriminator_subtract_gradients_b.push(create_matrix_matrix_subtract_function(1, discriminator_structure[layer]));
            discriminator_multiply_learning_rate_b.push(create_multiply_function(discriminator_structure[layer], 1, "scalar", STEP_SIZE));
        };

        let discriminator_slope = [];
        let discriminator_error = [];
        let discriminator_delta = [];
        let discriminator_w_gradient = [];
        let discriminator_b_gradient = [];
        for (let layer = 0; layer < discriminator_structure.length; layer++) {
            if (layer == (discriminator_structure.length - 1)) {
                discriminator_slope.push(create_sigmoid_function(BATCH_SIZE * discriminator_structure[layer], true));;
                discriminator_error.push(create_division_function(BATCH_SIZE, discriminator_structure[layer], -1));
            } else {
                discriminator_slope.push(create_leaky_relu_function(BATCH_SIZE * discriminator_structure[layer], true));;
                discriminator_error.push(create_multiply_function(BATCH_SIZE, discriminator_structure[layer + 1], discriminator_structure[layer]));
            };
            discriminator_delta.push(create_multiply_function(BATCH_SIZE, discriminator_structure[layer], "matrix"));
            if (layer == 0) {
                discriminator_w_gradient.push(create_multiply_function(image_size, BATCH_SIZE, discriminator_structure[layer]));
            } else {
                discriminator_w_gradient.push(create_multiply_function(discriminator_structure[layer - 1], BATCH_SIZE, discriminator_structure[layer]));
            };
            discriminator_b_gradient.push(create_sum_function(BATCH_SIZE, discriminator_structure[layer]));
        };
        let discriminator_error_fake = create_subtract_scalar_function(BATCH_SIZE, discriminator_structure[discriminator_structure.length - 1], 1);
        let discriminator_generator_error = create_division_function(BATCH_SIZE, discriminator_structure[discriminator_structure.length - 1], 1);
        let add_epsilon = create_add_scalar_function(BATCH_SIZE, discriminator_structure[discriminator_structure.length - 1], EPSILON);

        let generator_slope = [];
        let generator_error = [];
        let generator_delta = [];
        let generator_w_gradient = [];
        let generator_b_gradient = [];
        for (let layer = 0; layer < generator_structure.length; layer++) {
            if (layer == (generator_structure.length - 1)) {
                generator_slope.push(create_tanh_function(BATCH_SIZE * generator_structure[layer], true));;
                generator_error.push(create_multiply_function(BATCH_SIZE, discriminator_structure[discriminator_structure.length - 1], generator_structure[layer]));
            } else {
                generator_slope.push(create_leaky_relu_function(BATCH_SIZE * generator_structure[layer], true));;
                generator_error.push(create_multiply_function(BATCH_SIZE, generator_structure[layer + 1], generator_structure[layer]));
            };
            generator_delta.push(create_multiply_function(BATCH_SIZE, generator_structure[layer], "matrix"));
            if (layer == 0) {
                generator_w_gradient.push(create_multiply_function(noise_dimensions, BATCH_SIZE, generator_structure[layer]));
            } else {
                generator_w_gradient.push(create_multiply_function(generator_structure[layer - 1], BATCH_SIZE, generator_structure[layer]));
            };
            generator_b_gradient.push(create_sum_function(BATCH_SIZE, generator_structure[layer]));
        };

        window.epoch = 0;
        window.interval = setInterval(function() {
            // Generate fake images
            let generator_x_no_activation = [];
            let generator_x = [];
            for (let layer = 0; layer < (generator_structure.length + 1); layer++) {
                if (layer == 0) {
                    generator_x.push(new_random_array(BATCH_SIZE * noise_dimensions, -1, 1));
                    generator_x_no_activation.push(generator_x[layer]);
                } else {
                    generator_x_no_activation.push(generator_add[layer - 1](generator_matmul[layer - 1](generator_x[layer - 1], generator_w[layer - 1].array), generator_b[layer - 1].array));
                    generator_x.push(generator_layer[layer - 1](generator_x_no_activation[layer]));
                };
            };

            // Classify fake images
            let discriminator_x_no_activation = [];
            let discriminator_x = [];
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
            let discriminator_slope_ = [];
            let discriminator_w_transpose = [];
            let discriminator_layer_transpose = [];
            let discriminator_error_ = [];
            let discriminator_delta_ = [];
            let discriminator_w_gradient_fake = [];
            let discriminator_b_gradient_fake = [];
            for (let layer = 0; layer < discriminator_structure.length; layer++) {
                if (layer == discriminator_structure.length - 1) {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x[layer + 1]));
                } else {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x_no_activation[layer + 1]));
                    discriminator_w_transpose.push(discriminator_w[layer + 1].transpose_matrix.array);
                };
                if (layer == 0) {
                    discriminator_layer_transpose.push(new Matrix(discriminator_x[layer], BATCH_SIZE, image_size).transpose_matrix.array);
                } else {
                    discriminator_layer_transpose.push(new Matrix(discriminator_x[layer], BATCH_SIZE, discriminator_structure[layer]).array);
                };
            };
            
            for (let layer = discriminator_structure.length - 1; layer >= 0; layer--) {
                if (layer == discriminator_structure.length - 1) {
                    let discriminator_error_1 = discriminator_error_fake(discriminator_x[discriminator_x.length - 1]);
                    let discriminator_error_2 = add_epsilon(discriminator_error_1);
                    discriminator_error_[layer] = discriminator_error[layer](discriminator_error_2);
                } else {
                    discriminator_error_[layer] = discriminator_error[layer](discriminator_delta_[layer + 1], discriminator_w_transpose[layer]);
                };
                discriminator_delta_[layer] = discriminator_delta[layer](discriminator_slope_[layer], discriminator_error_[layer]);
                discriminator_w_gradient_fake[layer] = discriminator_w_gradient[layer](discriminator_layer_transpose[layer], discriminator_delta_[layer]);
                discriminator_b_gradient_fake[layer] = discriminator_b_gradient[layer](discriminator_delta_[layer]);
            };
            let discriminator_fake_loss = discriminator_error_[discriminator_error_.length - 1];

            // Classify real images
            let batch = MNIST_DATASET.next_batch(BATCH_SIZE);

            discriminator_x_no_activation = [];
            discriminator_x = [];
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
            discriminator_slope_ = [];
            discriminator_w_transpose = [];
            discriminator_layer_transpose = [];
            discriminator_error_ = [];
            discriminator_delta_ = [];
            discriminator_w_gradient_real = [];
            discriminator_b_gradient_real = [];
            for (let layer = 0; layer < discriminator_structure.length; layer++) {
                if (layer == discriminator_structure.length - 1) {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x[layer + 1]));
                } else {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x_no_activation[layer + 1]));
                    discriminator_w_transpose.push(discriminator_w[layer + 1].transpose_matrix.array);
                };
                if (layer == 0) {
                    discriminator_layer_transpose.push(new Matrix(discriminator_x[layer], BATCH_SIZE, image_size).transpose_matrix.array);
                } else {
                    discriminator_layer_transpose.push(new Matrix(discriminator_x[layer], BATCH_SIZE, discriminator_structure[layer]).array);
                };
            };
            
            for (let layer = discriminator_structure.length - 1; layer >= 0; layer--) {
                if (layer == discriminator_structure.length - 1) {
                    let discriminator_error_1 = add_epsilon(discriminator_x[discriminator_x.length - 1]);
                    discriminator_error_[layer] = discriminator_error[layer](discriminator_error_1);
                } else {
                    discriminator_error_[layer] = discriminator_error[layer](discriminator_delta_[layer + 1], discriminator_w_transpose[layer]);
                };
                discriminator_delta_[layer] = discriminator_delta[layer](discriminator_slope_[layer], discriminator_error_[layer]);
                discriminator_w_gradient_real[layer] = discriminator_w_gradient[layer](discriminator_layer_transpose[layer], discriminator_delta_[layer]);
                discriminator_b_gradient_real[layer] = discriminator_b_gradient[layer](discriminator_delta_[layer]);
            };

            let discriminator_real_loss = discriminator_error_[discriminator_error_.length - 1];

            // Add discriminator gradients
            let discriminator_w_gradient_ = [];
            let discriminator_b_gradient_ = [];
            for (let layer = 0; layer < discriminator_structure.length; layer++) {
                discriminator_w_gradient_[layer] = discriminator_subtract_gradients_w[layer](discriminator_w_gradient_fake[layer], discriminator_w_gradient_real[layer]);
                discriminator_b_gradient_[layer] = discriminator_subtract_gradients_b[layer](discriminator_b_gradient_fake[layer], discriminator_b_gradient_real[layer]);
            };

            // Generate fake images
            generator_x_no_activation = [];
            generator_x = [];
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
            discriminator_slope_ = [];
            discriminator_w_transpose = [];
            discriminator_error_ = [];
            discriminator_delta_ = [];
            discriminator_w_gradient_fake = [];
            discriminator_b_gradient_fake = [];
            for (let layer = 0; layer < discriminator_structure.length; layer++) {
                if (layer == discriminator_structure.length - 1) {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x[layer + 1]));
                } else {
                    discriminator_slope_.push(discriminator_slope[layer](discriminator_x_no_activation[layer + 1]));
                    discriminator_w_transpose.push(discriminator_w[layer + 1].transpose_matrix.array);
                };
            };
            
            for (let layer = discriminator_structure.length - 1; layer >= 0; layer--) {
                if (layer == discriminator_structure.length - 1) {
                    let discriminator_error_1 = add_epsilon(discriminator_x[discriminator_x.length - 1]);
                    discriminator_error_[layer] = discriminator_generator_error(discriminator_error_1);
                } else {
                    discriminator_error_[layer] = discriminator_error[layer](discriminator_delta_[layer + 1], discriminator_w_transpose[layer]);
                };
                discriminator_delta_[layer] = discriminator_delta[layer](discriminator_slope_[layer], discriminator_error_[layer]);
            };

            // Backprogate generator loss
            let generator_slope_ = [];
            let generator_w_transpose = [];
            let generator_layer_transpose = [];
            let generator_error_ = [];
            let generator_delta_ = [];
            let generator_w_gradient_ = [];
            let generator_b_gradient_ = [];
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
                    generator_error_[layer] = generator_error[layer](discriminator_delta_[0], generator_w_transpose[layer]);
                } else {
                    generator_error_[layer] = generator_error[layer](generator_delta_[layer + 1], generator_w_transpose[layer]);
                };
                generator_delta_[layer] = generator_delta[layer](generator_slope_[layer], generator_error_[layer]);
                generator_w_gradient_[layer] = generator_w_gradient[layer](generator_layer_transpose[layer], generator_delta_[layer]);
                generator_b_gradient_[layer] = generator_b_gradient[layer](generator_delta_[layer]);
            };

            let generator_loss = discriminator_error_[discriminator_error_.length - 1];

             // Optimize generator
            for (let layer = 0; layer < generator_structure.length; layer++) {
                if (layer == 0) {
                    generator_w[layer] = new Matrix(generator_subtract_gradients_w[layer](generator_w[layer].array, generator_multiply_learning_rate_w[layer](generator_w_gradient_[layer])), image_size, generator_structure[layer]);
                } else {
                    generator_w[layer] = new Matrix(generator_subtract_gradients_w[layer](generator_w[layer].array, generator_multiply_learning_rate_w[layer](generator_w_gradient_[layer])), generator_structure[layer - 1], generator_structure[layer]);
                };
                generator_b[layer] = new Vector(generator_subtract_gradients_b[layer](generator_b[layer].array, generator_multiply_learning_rate_b[layer](generator_b_gradient_[layer])));
            };

            // Optimize discriminator
            for (let layer = 0; layer < discriminator_structure.length; layer++) {
                if (layer == 0) {
                    discriminator_w[layer] = new Matrix(discriminator_subtract_gradients_w[layer](discriminator_w[layer].array, discriminator_multiply_learning_rate_w[layer](discriminator_w_gradient_[layer])), image_size, discriminator_structure[layer]);
                } else {
                    discriminator_w[layer] = new Matrix(discriminator_subtract_gradients_w[layer](discriminator_w[layer].array, discriminator_multiply_learning_rate_w[layer](discriminator_w_gradient_[layer])), discriminator_structure[layer - 1], discriminator_structure[layer]);
                };
                discriminator_b[layer] = new Vector(discriminator_subtract_gradients_b[layer](discriminator_b[layer].array, discriminator_multiply_learning_rate_b[layer](discriminator_b_gradient_[layer])));
            };


            console.log("Epoch: " + epoch + " Discriminator error real: " + new Matrix(discriminator_real_loss, discriminator_structure[discriminator_structure.length - 1], BATCH_SIZE).mean);
            console.log("Epoch: " + epoch + " Discriminator error fake: " + new Matrix(discriminator_fake_loss, discriminator_structure[discriminator_structure.length - 1], BATCH_SIZE).mean);
            console.log("Epoch: " + epoch + " Generator error: " + new Matrix(generator_loss, generator_structure[generator_structure.length - 1], BATCH_SIZE).mean);
            console.log("Epoch: " + epoch + " Learning rate: " + STEP_SIZE);

            if ((window.epoch + 1) % 100 == 0) {
                STEP_SIZE *= 1 / (1 + decay * (((window.epoch + 1) - ((window.epoch + 1) % 100)) / 100));
            };

            epoch_callback(batch/*generator_x[generator_x.length - 1]*/, new Matrix(discriminator_real_loss, BATCH_SIZE, 1).mean_squared_error, new Matrix(discriminator_fake_loss, BATCH_SIZE, 1).mean_squared_error, new Matrix(generator_loss, BATCH_SIZE, 1)[0, 0])
            window.epoch++;
        }, 10);
    });
};
function stop_training() {
    clearInterval(window.interval);
};