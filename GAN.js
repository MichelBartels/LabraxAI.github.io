function GAN(epoch_callback) {
    const BATCH_SIZE = 100;
    const NUMBER_OF_EPOCHS = 10000;
    const LEARNING_RATE_DISCRIMINATOR_REAL = 0.00009;
    const LEARNING_RATE_DISCRIMINATOR_FAKE = 0.00009;
    const LEARNING_RATE_GENERATOR = 0.00009;
    const noise_dimensions = 100;
    const generator_neurons_layer_1 = 112;
    const generator_neurons_layer_2 = 224;
    const generator_neurons_layer_3 = 336;
    const generator_neurons_layer_4 = 448;
    const generator_neurons_layer_5 = 560;
    const generator_neurons_layer_6 = 672;
    const generator_neurons_layer_7 = 784;
    const generator_structure = [112, 224, 336, 448,  560, 672, 784];
    const discriminator_structure = [128, 1];

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
                generator_w.push(new Matrix(new_random_array(noise_dimensions * generator_structure[layer]), noise_dimensions, generator_structure[layer]));
                generator_matmul.push(create_multiply_function(BATCH_SIZE, noise_dimensions, generator_structure[layer]));
            } else {
                generator_w.push(new Matrix(new_random_array(generator_structure[layer - 1] * generator_structure[layer]), generator_structure[layer - 1], generator_structure[layer]));
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

        let discriminator_w1 = new Matrix(new_random_array(784 * 128, -1, 1), 784, 128);
        let discriminator_matmul1 = create_multiply_function(BATCH_SIZE, 784, 128);
        let discriminator_b1 = new Vector(128, 0);
        let discriminator_add1 = create_matrix_vector_add_function(BATCH_SIZE, 128);
        let discriminator_layer1 = create_relu_function(BATCH_SIZE * 128);
        let discriminator_w2 = new Matrix(new_random_array(BATCH_SIZE * 128, -0.1, 0.1), 128, 1);
        let discriminator_matmul2 = create_multiply_function(BATCH_SIZE, 128, 1);
        let discriminator_b2 = new Vector(1, 0);
        let discriminator_add2 = create_matrix_vector_add_function(BATCH_SIZE, 1);
        let discriminator_output = create_tanh_function(BATCH_SIZE);
        let discriminator_error = create_matrix_matrix_subtract_function(BATCH_SIZE, 1);

        let discriminator_slope_output_layer = create_tanh_function(BATCH_SIZE * 1, true);
        let discriminator_slope_hidden_layer = create_relu_function(BATCH_SIZE * 128, true);
        let discriminator_derivative_output = create_multiply_function(BATCH_SIZE, 1, "matrix");
        let discriminator_error_at_hidden_layer = create_multiply_function(BATCH_SIZE, 1, 128);
        let discriminator_derivative_hidden_layer = create_multiply_function(BATCH_SIZE, 128, "matrix");

        let discriminator_w2_adjustments = create_multiply_function(128, BATCH_SIZE, 1);
        let discriminator_w2_adjustments_learning_rate_real = create_multiply_function(128, 1, "scalar", LEARNING_RATE_DISCRIMINATOR_REAL);
        let discriminator_w2_adjustments_learning_rate_fake = create_multiply_function(128, 1, "scalar", LEARNING_RATE_DISCRIMINATOR_FAKE);
        let discriminator_w2_apply_adjustments = create_matrix_matrix_subtract_function(128, 1);
        let discriminator_b2_adjustments = create_sum_function(BATCH_SIZE, 1);
        let discriminator_b2_adjustments_learning_rate_real = create_multiply_function(1, 1, "scalar", LEARNING_RATE_DISCRIMINATOR_REAL);
        let discriminator_b2_adjustments_learning_rate_fake = create_multiply_function(1, 1, "scalar", LEARNING_RATE_DISCRIMINATOR_FAKE);
        let discriminator_b2_apply_adjustments = create_matrix_matrix_subtract_function(1, 1);
        let discriminator_w1_adjustments = create_multiply_function(784, BATCH_SIZE, 128);
        let discriminator_w1_adjustments_learning_rate_real = create_multiply_function(784, 128, "scalar", LEARNING_RATE_DISCRIMINATOR_REAL);
        let discriminator_w1_adjustments_learning_rate_fake = create_multiply_function(784, 128, "scalar", LEARNING_RATE_DISCRIMINATOR_FAKE);
        let discriminator_w1_apply_adjustments = create_matrix_matrix_subtract_function(784, 128);
        let discriminator_b1_adjustments = create_sum_function(BATCH_SIZE, 128);
        let discriminator_b1_adjustments_learning_rate_real = create_multiply_function(128, 1, "scalar", LEARNING_RATE_DISCRIMINATOR_REAL);
        let discriminator_b1_adjustments_learning_rate_fake = create_multiply_function(128, 1, "scalar", LEARNING_RATE_DISCRIMINATOR_FAKE);
        let discriminator_b1_apply_adjustments = create_matrix_matrix_subtract_function(128, 1);

        let generator_slope = [];
        let generator_error = [];
        let generator_delta = [];
        let generator_w_adjustments = [];
        let generator_w_adjustments_learning_rate = [];
        let generator_w_apply_adjustments = [];
        let generator_b_adjustments = [];
        let generator_b_adjustments_learning_rate = [];
        let generator_b_apply_adjustments = [];
        for (let layer = 0; layer < generator_structure.length; layer++) {
            if (layer == (generator_structure.length - 1)) {
                generator_slope.push(create_tanh_function(BATCH_SIZE, generator_structure[layer]), true);;
                generator_error.push(BATCH_SIZE, discriminator_structure[discriminator_structure.length - 1], generator_structure[layer]);
            } else {
                generator_slope.push(create_relu_function(BATCH_SIZE, generator_structure[layer]), true);;
                generator_error.push(BATCH_SIZE, generator_structure[layer + 1], generator_structure[layer]);                
            };
            generator_delta.push(create_multiply_function(BATCH_SIZE, generator_structure[layer]));
            if (layer == 0) {
                generator_w_adjustments.push(create_multiply_function(noise_dimensions, BATCH_SIZE, generator_structure[layer]));
                generator_w_adjustments_learning_rate.push(create_multiply_function(noise_dimensions, generator_structure[layer], "scalar", LEARNING_RATE_GENERATOR));
                generator_w_apply_adjustments.push(create_matrix_matrix_subtract_function(noise_dimensions, generator_structure[layer]));
            } else {
                generator_w_adjustments.push(create_multiply_function(generator_structure[layer - 1], BATCH_SIZE, generator_structure[layer]));
                generator_w_adjustments_learning_rate.push(create_multiply_function(generator_structure[layer - 1], generator_structure[layer], "scalar", LEARNING_RATE_GENERATOR));
                generator_w_apply_adjustments.push(create_matrix_matrix_subtract_function(generator_structure[layer - 1], generator_structure[layer]))
            };
            generator_b_adjustments.push(create_sum_function(BATCH_SIZE, generator_structure[layer]));
            generator_b_adjustments_learning_rate.push(create_multiply_function(generator_structure[layer], 1, "scalar", LEARNING_RATE_GENERATOR))
            generator_b_apply_adjustments.push(create_matrix_matrix_subtract_function(generator_structure[layer], 1));
        };

        window.epoch = 0;
        window.interval = setInterval(function() {
            // Classify real images
            let batch = MNIST_DATASET.next_batch(BATCH_SIZE);
            
            let discriminator_layer_1_y_no_activation_real = discriminator_add1(discriminator_matmul1(batch, discriminator_w1.array), discriminator_b1.array);
            let discriminator_layer_1_y_real = discriminator_layer1(discriminator_layer_1_y_no_activation_real);
            let discriminator_y_real = discriminator_output(discriminator_add2(discriminator_matmul2(discriminator_layer_1_y_real, discriminator_w2.array), discriminator_b2.array));
            
            // Compute and print out discriminator error of real images
            let discriminator_y_real_targets = new Matrix(BATCH_SIZE, 1, 1);
            let discriminator_error_real = discriminator_error(discriminator_y_real, discriminator_y_real_targets.array);

            // Backpropagate discriminator error of real images
            let discriminator_slope_output_layer_real = discriminator_slope_output_layer(discriminator_y_real);
            let discriminator_slope_hidden_layer_real = discriminator_slope_hidden_layer(discriminator_layer_1_y_no_activation_real);
            let discriminator_derivative_output_real = discriminator_derivative_output(discriminator_slope_output_layer_real, discriminator_error_real);
            let discriminator_w2_transpose_ = new Matrix(discriminator_w2.array, 128, 1).transpose_matrix;
            let discriminator_error_at_hidden_layer_real = discriminator_error_at_hidden_layer(discriminator_derivative_output_real, discriminator_w2_transpose_.array);
            let discriminator_derivative_hidden_layer_real = discriminator_derivative_hidden_layer(discriminator_slope_hidden_layer_real, discriminator_error_at_hidden_layer_real);
            let discriminator_layer_1_y_transpose_real = new Matrix(discriminator_layer_1_y_real, BATCH_SIZE, 128).transpose_matrix;
            let discriminator_x_transpose_real = new Matrix(batch, BATCH_SIZE, 784).transpose_matrix;

            let discriminator_w2_adjustments_real = discriminator_w2_adjustments(discriminator_layer_1_y_transpose_real.array, discriminator_derivative_output_real);
            let discriminator_b2_adjustments_real = discriminator_b2_adjustments(discriminator_derivative_output_real);
            let discriminator_w2_adjustments_learning_rate_real_ = discriminator_w2_adjustments_learning_rate_real(discriminator_w2_adjustments_real);
            let discriminator_b2_adjustments_learning_rate_real_ = discriminator_b2_adjustments_learning_rate_real(discriminator_b2_adjustments_real);
            let discriminator_w1_adjustments_real = discriminator_w1_adjustments(discriminator_x_transpose_real.array, discriminator_derivative_hidden_layer_real);
            let discriminator_b1_adjustments_real = discriminator_b1_adjustments(discriminator_derivative_hidden_layer_real);
            let discriminator_w1_adjustments_learning_rate_real_ = discriminator_w1_adjustments_learning_rate_real(discriminator_w1_adjustments_real);
            let discriminator_b1_adjustments_learning_rate_real_ = discriminator_b1_adjustments_learning_rate_real(discriminator_b1_adjustments_real);
            
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
            let discriminator_layer_1_y_no_activation_fake = discriminator_add1(discriminator_matmul1(generator_x[generator_x.length - 1], discriminator_w1.array), discriminator_b1.array);
            let discriminator_layer_1_y_fake = discriminator_layer1(discriminator_layer_1_y_no_activation_fake);
            let discriminator_y_fake = discriminator_output(discriminator_add2(discriminator_matmul2(discriminator_layer_1_y_fake, discriminator_w2.array), discriminator_b2.array));

            // Compute and print out discriminator error of fake images
            let discriminator_y_fake_targets = new Matrix(BATCH_SIZE, 1, -1);
            let discriminator_error_fake = discriminator_error(discriminator_y_fake, discriminator_y_fake_targets.array);

            // Backpropagate discriminator error of fake images
            let discriminator_slope_output_layer_fake = discriminator_slope_output_layer(discriminator_y_fake);
            let discriminator_slope_hidden_layer_fake = discriminator_slope_hidden_layer(discriminator_layer_1_y_no_activation_fake);
            let discriminator_derivative_output_fake = discriminator_derivative_output(discriminator_slope_output_layer_fake, discriminator_error_fake);
            let discriminator_error_at_hidden_layer_fake = discriminator_error_at_hidden_layer(discriminator_derivative_output_fake, discriminator_w2_transpose_.array);
            let discriminator_derivative_hidden_layer_fake = discriminator_derivative_hidden_layer(discriminator_slope_hidden_layer_fake, discriminator_error_at_hidden_layer_fake);
            let discriminator_layer_1_y_transpose_fake = new Matrix(discriminator_layer_1_y_fake, BATCH_SIZE, 128).transpose_matrix;
            let discriminator_x_transpose_fake = new Matrix(generator_x[generator_x.length - 1], BATCH_SIZE, 784).transpose_matrix;

            let discriminator_w2_adjustments_fake = discriminator_w2_adjustments(discriminator_layer_1_y_transpose_fake.array, discriminator_derivative_output_fake);
            let discriminator_b2_adjustments_fake = discriminator_b2_adjustments(discriminator_derivative_output_fake);
            let discriminator_w2_adjustments_learning_rate_fake_ = discriminator_w2_adjustments_learning_rate_fake(discriminator_w2_adjustments_fake);
            let discriminator_b2_adjustments_learning_rate_fake_ = discriminator_b2_adjustments_learning_rate_fake(discriminator_b2_adjustments_fake);
            let discriminator_w1_adjustments_fake = discriminator_w1_adjustments(discriminator_x_transpose_fake.array, discriminator_derivative_hidden_layer_fake);
            let discriminator_b1_adjustments_fake = discriminator_b1_adjustments(discriminator_derivative_hidden_layer_fake);
            let discriminator_w1_adjustments_learning_rate_fake_ = discriminator_w1_adjustments_learning_rate_fake(discriminator_w1_adjustments_fake);
            let discriminator_b1_adjustments_learning_rate_fake_ = discriminator_b1_adjustments_learning_rate_fake(discriminator_b1_adjustments_fake);

            // Generate fake images
            generator_x_no_activation = [];
            generator_x = [];
            for (let layer = 0; layer < (generator_structure.length + 1); layer++) {
                if (layer == 0) {
                    generator_x_no_activation.push(new_random_array(BATCH_SIZE * noise_dimensions, -1, 1));
                    generator_x.push(generator_x_no_activation[layer]);
                } else {
                    generator_x_no_activation.push(generator_add[layer - 1](generator_matmul[layer - 1](generator_x[layer - 1], generator_w[layer - 1].array), generator_b[layer - 1].array));
                    generator_x.push(generator_layer[layer - 1](generator_x_no_activation[layer]));
                };
            };

            // Classify fake images
            discriminator_layer_1_y_no_activation_fake = discriminator_add1(discriminator_matmul1(generator_x[generator_x.length - 1], discriminator_w1.array), discriminator_b1.array);
            discriminator_layer_1_y_fake = discriminator_layer1(discriminator_layer_1_y_no_activation_fake);
            discriminator_y_fake = discriminator_output(discriminator_add2(discriminator_matmul2(discriminator_layer_1_y_fake, discriminator_w2.array), discriminator_b2.array));

            // Compute and print out generator error of fake images
            let generator_y_targets = new Matrix(BATCH_SIZE, 1, 1);
            let generator_error = discriminator_error(discriminator_y_fake, generator_y_targets.array);

            // Backpropagate generator error of fake images
            let generator_discriminator_slope_output_layer_fake = discriminator_slope_output_layer(discriminator_y_fake);
            let generator_discriminator_slope_hidden_layer_fake = discriminator_slope_hidden_layer(discriminator_layer_1_y_no_activation_fake);
            let generator_discriminator_derivative_output_fake = discriminator_derivative_output(generator_discriminator_slope_output_layer_fake, generator_error);
            let generator_discriminator_error_at_hidden_layer_fake = discriminator_error_at_hidden_layer(generator_discriminator_derivative_output_fake, discriminator_w2_transpose_.array);
            let generator_discriminator_derivative_hidden_layer_fake = discriminator_derivative_hidden_layer(generator_discriminator_slope_hidden_layer_fake, generator_discriminator_error_at_hidden_layer_fake);

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
                    generator_w_transpose.push(new Matrix(discriminator_w1_transpose_, generator_structure[layer], generator_structure[layer + 1]).transpose_matrix.array);
                } else {
                    generator_slope_.push(generator_slope[layer](generator_x_no_activation[layer + 1]));
                    generator_w_transpose.push(new Matrix(generator_w[layer + 1], generator_structure[layer], generator_structure[layer + 1]).transpose_matrix.array);
                };
                if (layer == 0) {
                    generator_layer_transpose.push(new Matrix(noise, BATCH_SIZE, noise_dimensions).transpose_matrix.array);
                } else {
                    generator_layer_transpose.push(new Matrix(generator_layer_[layer - 1], BATCH_SIZE, generator_structure[layer + 1]).array);
                };
            };
            
            for (let layer = generator_structure.length - 1; layer >= 0; layer++) {
                if (layer == generator_structure.length - 1) {
                    generator_error_[layer] = generator_error[layer](generator_discriminator_derivative_hidden_layer_fake, generator_w_transpose[layer]);
                } else {
                    generator_error_[layer] = generator_error[layer](generator_error_[layer + 1], generator_w_transpose[layer]);
                };
                generator_delta_[layer] = generator_delta[layer](generator_slope_[layer], generator_error_[layer]);
                generator_w_adjustments_.push(generator_w_adjustments[layer](generator_layer_transpose[layer], generator_delta_[layer]));
                generator_w_adjustments_learning_rate_.push(generator_w_adjustments_learning_rate[layer](generator_w_adjustments_[layer]));
                generator_b_adjustments_.push(generator_b_adjustments[layer](generator_delta_[layer]));
                generator_b_adjustments_learning_rate_.push(generator_b_adjustments_learning_rate[layer](generator_b_adjustments_[layer]));
            };

            // Optimize discriminator for classifying real images
            discriminator_w2 = new Matrix(discriminator_w2_apply_adjustments(discriminator_w2.array, discriminator_w2_adjustments_learning_rate_real_), 128, 1);
            discriminator_b2 = new Vector(discriminator_b2_apply_adjustments(discriminator_b2.array, discriminator_b2_adjustments_learning_rate_real_));
            discriminator_w1 = new Matrix(discriminator_w1_apply_adjustments(discriminator_w1.array, discriminator_w1_adjustments_learning_rate_real_), 784, 128);
            discriminator_b1 = new Vector(discriminator_b1_apply_adjustments(discriminator_b1.array, discriminator_b1_adjustments_learning_rate_real_));

            // Optimize discriminator for classifying fake images
            discriminator_w2 = new Matrix(discriminator_w2_apply_adjustments(discriminator_w2.array, discriminator_w2_adjustments_learning_rate_fake_), 128, 1);
            discriminator_b2 = new Vector(discriminator_b2_apply_adjustments(discriminator_b2.array, discriminator_b2_adjustments_learning_rate_fake_));
            discriminator_w1 = new Matrix(discriminator_w1_apply_adjustments(discriminator_w1.array, discriminator_w1_adjustments_learning_rate_fake_), 784, 128);
            discriminator_b1 = new Vector(discriminator_b1_apply_adjustments(discriminator_b1.array, discriminator_b1_adjustments_learning_rate_fake_));

            // Optimize generator
            for (let layer = 0; layer < generator_structure.length; layer++) {
                if (layer == 0) {
                    generator_w[layer] = new Matrix(generator_w_apply_adjustments[layer](generator_w[layer].array, generator_w_adjustments_learning_rate_[layer]), noise_dimensions, generator_structure[layer]); 
                } else {
                    generator_w[layer] = new Matrix(generator_w_apply_adjustments[layer](generator_w[layer].array, generator_w_adjustments_learning_rate_[layer]), generator_structure[layer - 1], generator_structure[layer]);
                };
                generator_b[layer] = new Vector(generator_b_apply_adjustments[layer](generator_b[layer].array, generator_b_adjustments_learning_rate_[layer]));
            };

            console.log("Epoch: " + epoch + " Discriminator error real: " + new Matrix(discriminator_error_real, BATCH_SIZE, 1).mean_squared_error);
            console.log("Epoch: " + epoch + " Discriminator error fake: " + new Matrix(discriminator_error_fake, BATCH_SIZE, 1).mean_squared_error);
            console.log("Epoch: " + epoch + " Generator error: " + new Matrix(generator_error, BATCH_SIZE, 1).mean_squared_error);

            epoch_callback(generator_y, new Matrix(discriminator_error_real, BATCH_SIZE, 1).mean_squared_error, new Matrix(discriminator_error_fake, BATCH_SIZE, 1).mean_squared_error, new Matrix(generator_error, BATCH_SIZE, 1).mean_squared_error)
            window.epoch++;
        }, 10);
    });
};
function stop_training() {
    clearInterval(window.interval);
};