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

    console.log("Loading MNIST dataset");

    new MNIST(function (MNIST_DATASET) {
        console.log("MNIST dataset loaded");
        let generator_w1 = new Matrix(new_random_array(noise_dimensions * generator_neurons_layer_1, -1, 1), noise_dimensions, generator_neurons_layer_1);
        let generator_matmul1 = create_multiply_function(BATCH_SIZE, noise_dimensions, generator_neurons_layer_1);
        let generator_b1 = new Vector(generator_neurons_layer_1, 0);
        let generator_add1 = create_matrix_vector_add_function(BATCH_SIZE, generator_neurons_layer_1);
        let generator_layer1 = create_relu_function(BATCH_SIZE * generator_neurons_layer_1);
        let generator_w2 = new Matrix(new_random_array(generator_neurons_layer_1 * generator_neurons_layer_2, -0.1, 0.1), generator_neurons_layer_1, generator_neurons_layer_2);
        let generator_matmul2 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_1, generator_neurons_layer_2);
        let generator_b2 = new Vector(generator_neurons_layer_2);
        let generator_add2 = create_matrix_vector_add_function(BATCH_SIZE, generator_neurons_layer_2);
        let generator_layer2 = create_relu_function(BATCH_SIZE * generator_neurons_layer_2);
        let generator_w3 = new Matrix(new_random_array(generator_neurons_layer_2 * generator_neurons_layer_3, -0.1, 0.1), generator_neurons_layer_2, generator_neurons_layer_3);
        let generator_matmul3 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_2, generator_neurons_layer_3);
        let generator_b3 = new Vector(generator_neurons_layer_3);
        let generator_add3 = create_matrix_vector_add_function(BATCH_SIZE, generator_neurons_layer_3);
        let generator_layer3 = create_relu_function(generator_neurons_layer_3 * BATCH_SIZE);
        let generator_w4 = new Matrix(new_random_array(generator_neurons_layer_3 * generator_neurons_layer_4, -0.1, 0.1), generator_neurons_layer_3, generator_neurons_layer_4);
        let generator_matmul4 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_3, generator_neurons_layer_4);
        let generator_b4 = new Vector(generator_neurons_layer_4);
        let generator_add4 = create_matrix_vector_add_function(BATCH_SIZE, generator_neurons_layer_4);
        let generator_layer4 = create_relu_function(generator_neurons_layer_4 * BATCH_SIZE);
        let generator_w5 = new Matrix(new_random_array(generator_neurons_layer_4 * generator_neurons_layer_5, -0.1, 0.1), generator_neurons_layer_4, generator_neurons_layer_5);
        let generator_matmul5 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_4, generator_neurons_layer_5);
        let generator_b5 = new Vector(generator_neurons_layer_5);
        let generator_add5 = create_matrix_vector_add_function(BATCH_SIZE, generator_neurons_layer_5);
        let generator_layer5 = create_relu_function(generator_neurons_layer_5 * BATCH_SIZE);
        let generator_w6 = new Matrix(new_random_array(generator_neurons_layer_5 * generator_neurons_layer_6, -0.1, 0.1), generator_neurons_layer_5, generator_neurons_layer_6);
        let generator_matmul6 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_5, generator_neurons_layer_6);
        let generator_b6 = new Vector(generator_neurons_layer_6);
        let generator_add6 = create_matrix_vector_add_function(BATCH_SIZE, generator_neurons_layer_6);
        let generator_layer6 = create_relu_function(generator_neurons_layer_6 * BATCH_SIZE);
        let generator_w7 = new Matrix(new_random_array(generator_neurons_layer_6 * generator_neurons_layer_7, -0.1, 0.1), generator_neurons_layer_6, generator_neurons_layer_7);
        let generator_matmul7 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_6, generator_neurons_layer_7);
        let generator_b7 = new Vector(generator_neurons_layer_7);
        let generator_add7 = create_matrix_vector_add_function(BATCH_SIZE, generator_neurons_layer_7);
        let generator_output = create_tanh_function(generator_neurons_layer_7 * BATCH_SIZE);

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

        let generator_slope_output_layer = create_relu_function(BATCH_SIZE * generator_neurons_layer_7, true);
        let generator_slope_layer_6 = create_relu_function(BATCH_SIZE * generator_neurons_layer_6, true);
        let generator_slope_layer_5 = create_relu_function(BATCH_SIZE * generator_neurons_layer_5, true);
        let generator_slope_layer_4 = create_relu_function(BATCH_SIZE * generator_neurons_layer_4, true);
        let generator_slope_layer_3 = create_relu_function(BATCH_SIZE * generator_neurons_layer_3, true);
        let generator_slope_layer_2 = create_relu_function(BATCH_SIZE * generator_neurons_layer_2, true);
        let generator_slope_layer_1 = create_relu_function(BATCH_SIZE * generator_neurons_layer_1, true);
        let generator_error_at_output_layer = create_multiply_function(BATCH_SIZE, 128, generator_neurons_layer_7);
        let generator_derivative_output_layer = create_multiply_function(BATCH_SIZE, generator_neurons_layer_7, "matrix");
        let generator_error_at_layer_6 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_7, generator_neurons_layer_6);
        let generator_derivative_layer_6 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_6, "matrix");
        let generator_error_at_layer_5 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_6, generator_neurons_layer_5);
        let generator_derivative_layer_5 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_5, "matrix");
        let generator_error_at_layer_4 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_5, generator_neurons_layer_4);
        let generator_derivative_layer_4 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_4, "matrix");
        let generator_error_at_layer_3 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_4, generator_neurons_layer_3);
        let generator_derivative_layer_3 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_3, "matrix");
        let generator_error_at_layer_2 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_3, generator_neurons_layer_2);
        let generator_derivative_layer_2 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_2, "matrix");
        let generator_error_at_layer_1 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_2, generator_neurons_layer_1);
        let generator_derivative_layer_1 = create_multiply_function(BATCH_SIZE, generator_neurons_layer_6, "matrix");

        let generator_w7_adjustments = create_multiply_function(generator_neurons_layer_6, BATCH_SIZE, generator_neurons_layer_7);
        let generator_w7_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_6, generator_neurons_layer_7, "scalar", LEARNING_RATE_GENERATOR);
        let generator_w7_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_6, generator_neurons_layer_7);
        let generator_b7_adjustments = create_sum_function(BATCH_SIZE, generator_neurons_layer_7);
        let generator_b7_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_7, 1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_b7_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_7, 1);
        let generator_w6_adjustments = create_multiply_function(256, BATCH_SIZE, generator_neurons_layer_6);
        let generator_w6_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_5, generator_neurons_layer_6, "scalar", LEARNING_RATE_GENERATOR);
        let generator_w6_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_5, generator_neurons_layer_6);
        let generator_b6_adjustments = create_sum_function(BATCH_SIZE, generator_neurons_layer_6);
        let generator_b6_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_6, 1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_b6_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_6, 1);
        let generator_w5_adjustments = create_multiply_function(generator_neurons_layer_4, BATCH_SIZE, generator_neurons_layer_5);
        let generator_w5_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_4, generator_neurons_layer_5, "scalar", LEARNING_RATE_GENERATOR);
        let generator_w5_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_4, generator_neurons_layer_5);
        let generator_b5_adjustments = create_sum_function(BATCH_SIZE, generator_neurons_layer_5);
        let generator_b5_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_5, 1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_b5_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_5, 1);
        let generator_w4_adjustments = create_multiply_function(generator_neurons_layer_3, BATCH_SIZE, generator_neurons_layer_4);
        let generator_w4_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_3, generator_neurons_layer_4, "scalar", LEARNING_RATE_GENERATOR);
        let generator_w4_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_3, generator_neurons_layer_4);
        let generator_b4_adjustments = create_sum_function(BATCH_SIZE, generator_neurons_layer_4);
        let generator_b4_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_4, 1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_b4_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_4, 1);
        let generator_w3_adjustments = create_multiply_function(generator_neurons_layer_2, BATCH_SIZE, generator_neurons_layer_3);
        let generator_w3_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_2, generator_neurons_layer_3, "scalar", LEARNING_RATE_GENERATOR);
        let generator_w3_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_2, generator_neurons_layer_3);
        let generator_b3_adjustments = create_sum_function(BATCH_SIZE, generator_neurons_layer_3);
        let generator_b3_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_3, 1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_b3_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_3, 1);
        let generator_w2_adjustments = create_multiply_function(generator_neurons_layer_1, BATCH_SIZE, generator_neurons_layer_2);
        let generator_w2_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_1, generator_neurons_layer_2, "scalar", LEARNING_RATE_GENERATOR);
        let generator_w2_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_1, generator_neurons_layer_2);
        let generator_b2_adjustments = create_sum_function(BATCH_SIZE, generator_neurons_layer_2);
        let generator_b2_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_2, 1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_b2_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_2, 1);
        let generator_w1_adjustments = create_multiply_function(noise_dimensions, BATCH_SIZE, generator_neurons_layer_1);
        let generator_w1_adjustments_learning_rate = create_multiply_function(noise_dimensions, generator_neurons_layer_1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_w1_apply_adjustments = create_matrix_matrix_subtract_function(noise_dimensions, generator_neurons_layer_1);
        let generator_b1_adjustments = create_sum_function(BATCH_SIZE, generator_neurons_layer_1);
        let generator_b1_adjustments_learning_rate = create_multiply_function(generator_neurons_layer_1, 1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_b1_apply_adjustments = create_matrix_matrix_subtract_function(generator_neurons_layer_1, 1);

        let epoch = 0;
        setInterval(function() {
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
            let noise = new_random_array(BATCH_SIZE * 100, -1, 1);
            let generator_layer_1_y_no_activation = generator_add1(generator_matmul1(noise, generator_w1.array), generator_b1.array);
            let generator_layer_1_y = generator_layer1(generator_layer_1_y_no_activation);
            let generator_layer_2_y_no_activation = generator_add2(generator_matmul2(generator_layer_1_y, generator_w2.array), generator_b2.array);
            let generator_layer_2_y = generator_layer2(generator_layer_2_y_no_activation);
            let generator_layer_3_y_no_activation = generator_add3(generator_matmul3(generator_layer_2_y, generator_w3.array), generator_b3.array);
            let generator_layer_3_y = generator_layer3(generator_layer_3_y_no_activation);
            let generator_layer_4_y_no_activation = generator_add4(generator_matmul4(generator_layer_3_y, generator_w4.array), generator_b4.array);
            let generator_layer_4_y = generator_layer4(generator_layer_4_y_no_activation);
            let generator_layer_5_y_no_activation = generator_add5(generator_matmul5(generator_layer_4_y, generator_w5.array), generator_b5.array);
            let generator_layer_5_y = generator_layer5(generator_layer_5_y_no_activation);
            let generator_layer_5_y_no_activation = generator_add5(generator_matmul5(generator_layer_4_y, generator_w5.array), generator_b5.array);
            let generator_layer_5_y = generator_layer5(generator_layer_5_y_no_activation);
            let generator_layer_6_y_no_activation = generator_add6(generator_matmul6(generator_layer_5_y, generator_w6.array), generator_b6.array);
            let generator_layer_6_y = generator_layer6(generator_layer_6_y_no_activation);
            let generator_y = generator_output(generator_add7(generator_matmul7(generator_layer_6_y, generator_w7.array), generator_b7.array));

            // Classify fake images
            let discriminator_layer_1_y_no_activation_fake = discriminator_add1(discriminator_matmul1(generator_y, discriminator_w1.array), discriminator_b1.array);
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
            let discriminator_x_transpose_fake = new Matrix(generator_y, BATCH_SIZE, 784).transpose_matrix;

            let discriminator_w2_adjustments_fake = discriminator_w2_adjustments(discriminator_layer_1_y_transpose_fake.array, discriminator_derivative_output_fake);
            let discriminator_b2_adjustments_fake = discriminator_b2_adjustments(discriminator_derivative_output_fake);
            let discriminator_w2_adjustments_learning_rate_fake_ = discriminator_w2_adjustments_learning_rate_fake(discriminator_w2_adjustments_fake);
            let discriminator_b2_adjustments_learning_rate_fake_ = discriminator_b2_adjustments_learning_rate_fake(discriminator_b2_adjustments_fake);
            let discriminator_w1_adjustments_fake = discriminator_w1_adjustments(discriminator_x_transpose_fake.array, discriminator_derivative_hidden_layer_fake);
            let discriminator_b1_adjustments_fake = discriminator_b1_adjustments(discriminator_derivative_hidden_layer_fake);
            let discriminator_w1_adjustments_learning_rate_fake_ = discriminator_w1_adjustments_learning_rate_fake(discriminator_w1_adjustments_fake);
            let discriminator_b1_adjustments_learning_rate_fake_ = discriminator_b1_adjustments_learning_rate_fake(discriminator_b1_adjustments_fake);

            // Generate fake images
            noise = new_random_array(BATCH_SIZE * 100, -1, 1);
            generator_layer_1_y_no_activation = generator_add1(generator_matmul1(noise, generator_w1.array), generator_b1.array);
            generator_layer_1_y = generator_layer1(generator_layer_1_y_no_activation);
            generator_layer_2_y_no_activation = generator_add2(generator_matmul2(generator_layer_1_y, generator_w2.array), generator_b2.array);
            generator_layer_2_y = generator_layer2(generator_layer_2_y_no_activation);
            generator_layer_3_y_no_activation = generator_add3(generator_matmul3(generator_layer_2_y, generator_w3.array), generator_b3.array);
            generator_layer_3_y = generator_layer3(generator_layer_3_y_no_activation);
            generator_layer_4_y_no_activation = generator_add4(generator_matmul4(generator_layer_3_y, generator_w4.array), generator_b4.array);
            generator_layer_4_y = generator_layer4(generator_layer_4_y_no_activation);
            generator_layer_5_y_no_activation = generator_add5(generator_matmul5(generator_layer_4_y, generator_w5.array), generator_b5.array);
            generator_layer_5_y = generator_layer5(generator_layer_5_y_no_activation);
            generator_layer_5_y_no_activation = generator_add5(generator_matmul5(generator_layer_4_y, generator_w5.array), generator_b5.array);
            generator_layer_5_y = generator_layer5(generator_layer_5_y_no_activation);
            generator_layer_6_y_no_activation = generator_add6(generator_matmul6(generator_layer_5_y, generator_w6.array), generator_b6.array);
            generator_layer_6_y = generator_layer6(generator_layer_6_y_no_activation);
            generator_y = generator_output(generator_add7(generator_matmul7(generator_layer_6_y, generator_w7.array), generator_b7.array));

            // Classify fake images
            discriminator_layer_1_y_no_activation_fake = discriminator_add1(discriminator_matmul1(generator_y, discriminator_w1.array), discriminator_b1.array);
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

            let generator_slope_output_layer_ = generator_slope_output_layer(generator_y);
            let generator_slope_layer_6_ = generator_slope_layer_2(generator_layer_6_y_no_activation);
            let generator_slope_layer_5_ = generator_slope_layer_1(generator_layer_5_y_no_activation);
            let generator_slope_layer_4_ = generator_slope_layer_2(generator_layer_4_y_no_activation);
            let generator_slope_layer_3_ = generator_slope_layer_1(generator_layer_3_y_no_activation);
            let generator_slope_layer_2_ = generator_slope_layer_2(generator_layer_2_y_no_activation);
            let generator_slope_layer_1_ = generator_slope_layer_1(generator_layer_1_y_no_activation);
            let discriminator_w1_transpose_ = new Matrix(discriminator_w1.array, generator_neurons_layer_7, 128).transpose_matrix;
            let generator_error_at_output_layer_ = generator_error_at_output_layer(generator_discriminator_derivative_hidden_layer_fake, discriminator_w1_transpose_.array);
            let generator_derivative_output_layer_ = generator_derivative_output_layer(generator_slope_output_layer_, generator_error_at_output_layer_);
            let generator_w7_transpose_ = new Matrix(generator_w3.array, generator_neurons_layer_6, generator_neurons_layer_7).transpose_matrix;
            let generator_error_at_layer_6_ = generator_error_at_layer_2(generator_derivative_output_layer_, generator_w3_transpose_.array);
            let generator_derivative_layer_6_ = generator_derivative_layer_2(generator_slope_layer_2_, generator_error_at_layer_2_);
            let generator_w6_transpose_ = new Matrix(generator_w3.array, generator_neurons_layer_5, generator_neurons_layer_6).transpose_matrix;
            let generator_error_at_layer_5_ = generator_error_at_layer_2(generator_derivative_output_layer_, generator_w3_transpose_.array);
            let generator_derivative_layer_5_ = generator_derivative_layer_2(generator_slope_layer_2_, generator_error_at_layer_2_);
            let generator_w5_transpose_ = new Matrix(generator_w3.array, generator_neurons_layer_4, generator_neurons_layer_5).transpose_matrix;
            let generator_error_at_layer_4_ = generator_error_at_layer_2(generator_derivative_output_layer_, generator_w3_transpose_.array);
            let generator_derivative_layer_4_ = generator_derivative_layer_2(generator_slope_layer_2_, generator_error_at_layer_2_);
            let generator_w4_transpose_ = new Matrix(generator_w3.array, generator_neurons_layer_3, generator_neurons_layer_4).transpose_matrix;
            let generator_error_at_layer_3_ = generator_error_at_layer_2(generator_derivative_output_layer_, generator_w3_transpose_.array);
            let generator_derivative_layer_3_ = generator_derivative_layer_2(generator_slope_layer_2_, generator_error_at_layer_2_);
            let generator_w3_transpose_ = new Matrix(generator_w3.array, generator_neurons_layer_2, generator_neurons_layer_3).transpose_matrix;
            let generator_error_at_layer_2_ = generator_error_at_layer_2(generator_derivative_output_layer_, generator_w3_transpose_.array);
            let generator_derivative_layer_2_ = generator_derivative_layer_2(generator_slope_layer_2_, generator_error_at_layer_2_);
            let generator_w2_transpose_ = new Matrix(generator_w2.array, generator_neurons_layer_1, generator_neurons_layer_2).transpose_matrix;
            let generator_error_at_layer_1_ = generator_error_at_layer_1(generator_derivative_layer_2_, generator_w2_transpose_.array);
            let generator_derivative_layer_1_ = generator_derivative_layer_1(generator_slope_layer_1_, generator_error_at_layer_1_);
            let generator_layer_6_y_transpose_ = new Matrix(generator_layer_6_y, BATCH_SIZE, generator_neurons_layer_6).transpose_matrix;
            let generator_layer_5_y_transpose_ = new Matrix(generator_layer_5_y, BATCH_SIZE, generator_neurons_layer_5).transpose_matrix;
            let generator_layer_4_y_transpose_ = new Matrix(generator_layer_4_y, BATCH_SIZE, generator_neurons_layer_4).transpose_matrix;
            let generator_layer_3_y_transpose_ = new Matrix(generator_layer_3_y, BATCH_SIZE, generator_neurons_layer_3).transpose_matrix;
            let generator_layer_2_y_transpose_ = new Matrix(generator_layer_2_y, BATCH_SIZE, generator_neurons_layer_2).transpose_matrix;
            let generator_layer_1_y_transpose_ = new Matrix(generator_layer_1_y, BATCH_SIZE, generator_neurons_layer_1).transpose_matrix;
            let generator_noise_transpose_ = new Matrix(noise, BATCH_SIZE, noise_dimensions).transpose_matrix;

            let generator_w7_adjustments_ = generator_w7_adjustments(generator_layer_6_y_transpose_.array, generator_derivative_output_layer_);
            let generator_b7_adjustments_ = generator_b7_adjustments(generator_derivative_output_layer_);
            let generator_w7_adjustments_learning_rate_ = generator_w7_adjustments_learning_rate(generator_w7_adjustments_);
            let generator_b7_adjustments_learning_rate_ = generator_b7_adjustments_learning_rate(generator_b7_adjustments_);
            let generator_w6_adjustments_ = generator_w6_adjustments(generator_layer_5_y_transpose_.array, generator_derivative_layer_6_);
            let generator_b6_adjustments_ = generator_b6_adjustments(generator_derivative_layer_2_);
            let generator_w6_adjustments_learning_rate_ = generator_w6_adjustments_learning_rate(generator_w6_adjustments_);
            let generator_b6_adjustments_learning_rate_ = generator_b6_adjustments_learning_rate(generator_b6_adjustments_);
            let generator_w5_adjustments_ = generator_w5_adjustments(generator_layer_4_y_transpose_.array, generator_derivative_layer_5_);
            let generator_b5_adjustments_ = generator_b5_adjustments(generator_derivative_output_layer_);
            let generator_w5_adjustments_learning_rate_ = generator_w5_adjustments_learning_rate(generator_w5_adjustments_);
            let generator_b5_adjustments_learning_rate_ = generator_b5_adjustments_learning_rate(generator_b5_adjustments_);
            let generator_w4_adjustments_ = generator_w4_adjustments(generator_layer_3_y_transpose_.array, generator_derivative_layer_4_);
            let generator_b4_adjustments_ = generator_b4_adjustments(generator_derivative_layer_2_);
            let generator_w4_adjustments_learning_rate_ = generator_w4_adjustments_learning_rate(generator_w4_adjustments_);
            let generator_b4_adjustments_learning_rate_ = generator_b4_adjustments_learning_rate(generator_b4_adjustments_);
            let generator_w3_adjustments_ = generator_w3_adjustments(generator_layer_2_y_transpose_.array, generator_derivative_layer_3_);
            let generator_b3_adjustments_ = generator_b3_adjustments(generator_derivative_output_layer_);
            let generator_w3_adjustments_learning_rate_ = generator_w3_adjustments_learning_rate(generator_w3_adjustments_);
            let generator_b3_adjustments_learning_rate_ = generator_b3_adjustments_learning_rate(generator_b3_adjustments_);
            let generator_w2_adjustments_ = generator_w2_adjustments(generator_layer_1_y_transpose_.array, generator_derivative_layer_2_);
            let generator_b2_adjustments_ = generator_b2_adjustments(generator_derivative_layer_2_);
            let generator_w2_adjustments_learning_rate_ = generator_w2_adjustments_learning_rate(generator_w2_adjustments_);
            let generator_b2_adjustments_learning_rate_ = generator_b2_adjustments_learning_rate(generator_b2_adjustments_); 
            let generator_w1_adjustments_ = generator_w1_adjustments(generator_noise_transpose_.array, generator_derivative_layer_1_);
            let generator_b1_adjustments_ = generator_b1_adjustments(generator_derivative_layer_1_);
            let generator_w1_adjustments_learning_rate_ = generator_w1_adjustments_learning_rate(generator_w1_adjustments_);
            let generator_b1_adjustments_learning_rate_ = generator_b1_adjustments_learning_rate(generator_b1_adjustments_);

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
            generator_w7 = new Matrix(generator_w1_apply_adjustments(generator_w7.array, generator_w7_adjustments_learning_rate_), 1, 256);
            generator_b7 = new Vector(generator_b1_apply_adjustments(generator_b7.array, generator_b7_adjustments_learning_rate_));
            generator_w6 = new Matrix(generator_w2_apply_adjustments(generator_w6.array, generator_w6_adjustments_learning_rate_), 256, 784);
            generator_b6 = new Vector(generator_b2_apply_adjustments(generator_b6.array, generator_b6_adjustments_learning_rate_));
            generator_w5 = new Matrix(generator_w2_apply_adjustments(generator_w5.array, generator_w5_adjustments_learning_rate_), 256, 256);
            generator_b5 = new Vector(generator_b2_apply_adjustments(generator_b5.array, generator_b5_adjustments_learning_rate_));
            generator_w4 = new Matrix(generator_w1_apply_adjustments(generator_w4.array, generator_w4_adjustments_learning_rate_), 1, 256);
            generator_b4 = new Vector(generator_b1_apply_adjustments(generator_b4.array, generator_b4_adjustments_learning_rate_));
            generator_w3 = new Matrix(generator_w2_apply_adjustments(generator_w3.array, generator_w3_adjustments_learning_rate_), 256, 784);
            generator_b3 = new Vector(generator_b2_apply_adjustments(generator_b3.array, generator_b3_adjustments_learning_rate_));
            generator_w2 = new Matrix(generator_w2_apply_adjustments(generator_w2.array, generator_w2_adjustments_learning_rate_), 256, 256);
            generator_b2 = new Vector(generator_b2_apply_adjustments(generator_b2.array, generator_b2_adjustments_learning_rate_));
            generator_w1 = new Matrix(generator_w1_apply_adjustments(generator_w1.array, generator_w1_adjustments_learning_rate_), 1, 256);
            generator_b1 = new Vector(generator_b1_apply_adjustments(generator_b1.array, generator_b1_adjustments_learning_rate_));

            console.log("Epoch: " + epoch + " Discriminator error real: " + new Matrix(discriminator_error_real, BATCH_SIZE, 1).mean_squared_error);
            console.log("Epoch: " + epoch + " Discriminator error fake: " + new Matrix(discriminator_error_fake, BATCH_SIZE, 1).mean_squared_error);
            console.log("Epoch: " + epoch + " Generator error: " + new Matrix(generator_error, BATCH_SIZE, 1).mean_squared_error);

            epoch_callback(generator_y, new Matrix(discriminator_error_real, BATCH_SIZE, 1).mean_squared_error, new Matrix(discriminator_error_fake, BATCH_SIZE, 1).mean_squared_error, new Matrix(generator_error, BATCH_SIZE, 1).mean_squared_error)
            epoch++;
        }, 10);
    });
};