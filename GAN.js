function GAN(epoch_callback) {
    const BATCH_SIZE = 100;
    const NUMBER_OF_EPOCHS = 10000;
    const LEARNING_RATE_DISCRIMINATOR_REAL = 0.001;
    const LEARNING_RATE_DISCRIMINATOR_FAKE = 0.001;
    const LEARNING_RATE_GENERATOR = 0.001;

    console.log("Loading MNIST dataset");

    new MNIST(function (MNIST_DATASET) {
        console.log("MNIST dataset loaded");
        let generator_w1 = new Matrix(new_random_array(100 * 128, -1, 1), 100, 128);
        let generator_matmul1 = create_multiply_function(BATCH_SIZE, 100, 128);
        let generator_b1 = new Vector(128, 0);
        let generator_add1 = create_matrix_vector_add_function(100, 128);
        let generator_layer1 = create_relu_function(BATCH_SIZE * 128);
        let generator_w2 = new Matrix(new_random_array(100 * 128, -0.1, 0.1), 128, 784);
        let generator_matmul2 = create_multiply_function(BATCH_SIZE, 128, 784);
        let generator_b2 = new Vector(784);
        let generator_add2 = create_matrix_vector_add_function(BATCH_SIZE, 784);
        let generator_output = create_tanh_function(784 * BATCH_SIZE);

        let discriminator_w1 = new Matrix(new_random_array(784 * 128, -1, 1), 784, 128);
        let discriminator_matmul1 = create_multiply_function(BATCH_SIZE, 784, 128);
        let discriminator_b1 = new Vector(128, 0);
        let discriminator_add1 = create_matrix_vector_add_function(100, 128);
        let discriminator_layer1 = create_relu_function(BATCH_SIZE * 128);
        let discriminator_w2 = new Matrix(new_random_array(100 * 128, -0.1, 0.1), 128, 1);
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

        let generator_slope_output_layer = create_relu_function(BATCH_SIZE * 784, true);
        let generator_slope_hidden_layer = create_relu_function(BATCH_SIZE * 128, true);
        let generator_error_at_output_layer = create_multiply_function(BATCH_SIZE, 128, 784);
        let generator_derivative_output_layer = create_multiply_function(BATCH_SIZE, 784, "matrix");
        let generator_error_at_hidden_layer = create_multiply_function(BATCH_SIZE, 784, 128);
        let generator_derivative_hidden_layer = create_multiply_function(BATCH_SIZE, 128, "matrix");

        let generator_w2_adjustments = create_multiply_function(128, BATCH_SIZE, 784);
        let generator_w2_adjustments_learning_rate = create_multiply_function(128, 784, "scalar", LEARNING_RATE_GENERATOR);
        let generator_w2_apply_adjustments = create_matrix_matrix_subtract_function(128, 784);
        let generator_b2_adjustments = create_sum_function(BATCH_SIZE, 784);
        let generator_b2_adjustments_learning_rate = create_multiply_function(784, 1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_b2_apply_adjustments = create_matrix_matrix_subtract_function(784, 1);
        let generator_w1_adjustments = create_multiply_function(100, BATCH_SIZE, 128);
        let generator_w1_adjustments_learning_rate = create_multiply_function(100, 128, "scalar", LEARNING_RATE_GENERATOR);
        let generator_w1_apply_adjustments = create_matrix_matrix_subtract_function(100, 128);
        let generator_b1_adjustments = create_sum_function(BATCH_SIZE, 128);
        let generator_b1_adjustments_learning_rate = create_multiply_function(128, 1, "scalar", LEARNING_RATE_GENERATOR);
        let generator_b1_apply_adjustments = create_matrix_matrix_subtract_function(128, 1);

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
            let generator_y = generator_output(generator_add2(generator_matmul2(generator_layer_1_y, generator_w2.array), generator_b2.array));

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
            generator_y = generator_output(generator_add2(generator_matmul2(generator_layer_1_y, generator_w2.array), generator_b2.array));

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
            let generator_slope_hidden_layer_ = generator_slope_hidden_layer(generator_layer_1_y_no_activation);
            let discriminator_w1_transpose_ = new Matrix(discriminator_w1.array, 784, 128).transpose_matrix;
            let generator_error_at_output_layer_ = generator_error_at_output_layer(generator_discriminator_derivative_hidden_layer_fake, discriminator_w1_transpose_.array);
            let generator_derivative_output_layer_ = generator_derivative_output_layer(generator_slope_output_layer_, generator_error_at_output_layer_);
            let generator_w2_transpose_ = new Matrix(generator_w2.array, 128, 784).transpose_matrix;
            let generator_error_at_hidden_layer_ = generator_error_at_hidden_layer(generator_derivative_output_layer_, generator_w2_transpose_.array);
            let generator_derivative_hidden_layer_ = generator_derivative_hidden_layer(generator_slope_hidden_layer_, generator_error_at_hidden_layer_);
            let generator_layer_1_y_transpose_ = new Matrix(generator_layer_1_y, BATCH_SIZE, 128).transpose_matrix;
            let generator_noise_transpose_ = new Matrix(noise, BATCH_SIZE, 100).transpose_matrix;

            let generator_w2_adjustments_ = generator_w2_adjustments(generator_layer_1_y_transpose_.array, generator_derivative_output_layer_);
            let generator_b2_adjustments_ = generator_b2_adjustments(generator_derivative_output_layer_);
            let generator_w2_adjustments_learning_rate_ = generator_w2_adjustments_learning_rate(generator_w2_adjustments_);
            let generator_b2_adjustments_learning_rate_ = generator_b2_adjustments_learning_rate(generator_b2_adjustments_);
            let generator_w1_adjustments_ = generator_w1_adjustments(generator_noise_transpose_.array, generator_derivative_hidden_layer_);
            let generator_b1_adjustments_ = generator_b1_adjustments(generator_derivative_hidden_layer_);
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
            generator_w2 = new Matrix(generator_w2_apply_adjustments(generator_w2.array, generator_w2_adjustments_learning_rate_), 128, 784);
            generator_b2 = new Vector(generator_b2_apply_adjustments(generator_b2.array, generator_b2_adjustments_learning_rate_));
            generator_w1 = new Matrix(generator_w1_apply_adjustments(generator_w1.array, generator_w1_adjustments_learning_rate_), 1, 128);
            generator_b1 = new Vector(generator_b1_apply_adjustments(generator_b1.array, generator_b1_adjustments_learning_rate_));

            console.log("Epoch: " + epoch + " Discriminator error real: " + new Matrix(discriminator_error_real, BATCH_SIZE, 1).mean_squared_error);
            console.log("Epoch: " + epoch + " Discriminator error fake: " + new Matrix(discriminator_error_fake, BATCH_SIZE, 1).mean_squared_error);
            console.log("Epoch: " + epoch + " Generator error: " + new Matrix(generator_error, BATCH_SIZE, 1).mean_squared_error);

            epoch_callback(generator_y, new Matrix(discriminator_error_real, BATCH_SIZE, 1).mean_squared_error, new Matrix(discriminator_error_fake, BATCH_SIZE, 1).mean_squared_error, new Matrix(generator_error, BATCH_SIZE, 1).mean_squared_error)
            epoch++;
        }, 10);
    });
};