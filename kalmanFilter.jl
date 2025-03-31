# using Pkg
# Pkg.add(["LinearAlgebra", "Plots", "DataFrames", "GLM", "StatsModels", "Statistics"])
using LinearAlgebra
using Plots # To graph points
using GLM, DataFrames, StatsModels, Statistics

points = rand(Float64, 16,2) * 10
x = points[:,1]
y = points[:,2]

# =====================================
# CORE COMPUTATION FUNCTIONS (No plotting)
# =====================================

function linearRegression_compute(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Limit to n points
    x_subset = x[1:n]
    A = hcat(ones(n), x_subset)
    y_subset = y[1:n]
    A_inv = inv(A' * A)
    x_hat = A_inv * (A' * y_subset)
    return A_inv, x_hat
end

function recursiveRegression_compute(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Initial regression with first n points
    covar, x_hat = linearRegression_compute(x, y, n)

    for iter in n+1:length(x)
        x_new = [1; x[iter]]  # Column vector
        y_new = y[iter]
        
        gain_vector = covar * x_new
        c = 1 / (1 + (x_new' * gain_vector))
        covar = covar - c * gain_vector * x_new' * covar
        error = y_new - (x_new' * x_hat)

        x_hat = x_hat + c * error * gain_vector
    end
    return covar, x_hat
end

function covariance_matrix(variances::Vector{Float64}, covariances::Vector{Float64})
    n = length(variances)
    d = Diagonal(variances)
    z = zeros(n, n)
    counter = 1
    for i in 2:n
        for j in 1:i-1
            z[i,j] = covariances[counter]
            counter += 1
        end
    end
    covar = Symmetric(d + z, :L)
    # Be mind that setting type to matrices improves matrices performance
    # Julia optimize calculations based on matrix structure
    return covar
end

# Overloading when no parameters are passed
function covariance_matrix(n::Int64, var::Float64=1.0, covar::Float64=0.5)
    variances = ones(n) * var
    # Factorial sum
    n_reduced = n - 1
    triangular_num_fun = n_reduced * (n_reduced + 1) ÷ 2 # \div mean integer division
    covariances = ones(triangular_num_fun) * covar  
    return covariance_matrix(variances, covariances)
end

function CovarRecurrentRegression_compute(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    covar, x_hat = linearRegression_compute(x, y, n)
    
    x_subset = x[n+1:end]
    len = length(x_subset)
    current_A = hcat(ones(len), x_subset)
    y_subset = y[n+1:end]

    inv_measurement_covar = covariance_matrix(len,1.0,0.0)^-1
    error_covar = covar^-1  # In this case we suppose that the initial covariance matrix for the previous points was the identity

    error_covar = error_covar + (current_A' * inv_measurement_covar * current_A)

    #kalman_gain = (error_covar^-1) * current_A' * inv_measurement_covar
    #To avoid calculating its inverse, we express it like a equation
    #This way we essentially only do one inverse (the first time after doing normal least squares)
    kalman_gain = error_covar \ (current_A' * inv_measurement_covar)

    x_hat = x_hat + kalman_gain * (y_subset - current_A * x_hat) 
    return error_covar, x_hat
end

# function KalmanRecurrentRegression_compute(x::Vector{Float64}, y::Vector{Float64}, n::Int)
#     # Initial regression with first n points
#     covar, x_hat = linearRegression_compute(x, y, n)
    
#     measurement_var = 1.0  # Measurement noise variance
    
#     for iter in n+1:length(x)
#         x_new = [1; x[iter]]  # Column vector
#         y_new = y[iter]
        
#         # Calculate Kalman gain
#         kalman_gain = covar * x_new / (x_new' * covar * x_new + measurement_var)
        
#         # Update state estimate
#         error = y_new - (x_new' * x_hat)
#         x_hat = x_hat + kalman_gain * error[1]
        
#         # Update covariance matrix (using Joseph form for numerical stability)
#         covar = (I - kalman_gain * x_new') * covar
#     end
    
#     return covar, x_hat
# end

# =====================================
# PLOTTING FUNCTIONS
# =====================================

function linearRegression_plot(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Get the computation result
    covar, x_hat = linearRegression_compute(x, y, n)
    
    # Get the first n points for plotting
    x_subset = x[1:n]
    y_subset = y[1:n]
    
    p = scatter(x_subset, y_subset, label="Data", title="Linear Regression (first $n points)", 
                markersize=5, color=:blue, markerstrokewidth=0)
    plot!(p, x_subset, x_hat[1] .+ x_hat[2] .* x_subset, label="Regression", 
          color=:green, linewidth=2)
    
    return covar, x_hat, p 
end

function recursiveRegression_plot(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Initial regression with first n points
    covar, x_hat = linearRegression_compute(x, y, n)
    
    # Create first plot for initial fit
    p1 = scatter(x[1:n], y[1:n], label="Data", title="Linear Regression (first $n points)", 
                markersize=5, color=:blue, markerstrokewidth=0)
    plot!(p1, x[1:n], x_hat[1] .+ x_hat[2] .* x[1:n], label="Regression", 
          color=:green, linewidth=2)

    len = length(x)

    p2 = plot(title="Recursive Least Squares Evolution", legend=:outertopright)
    scatter!(p2, x[1:n], y[1:n], label="Training Data (1-$n)", 
             markersize=5, color=:blue, markerstrokewidth=0)
    
    # New points (n+1 to end) in orange
    scatter!(p2, x[n+1:end], y[n+1:end], label="New Data ($(n+1)-$len)", 
             markersize=5, color=:orange, markershape=:diamond, markerstrokewidth=0)

    # Plot initial model
    initial_x = range(minimum(x), maximum(x), length=100)
    initial_line = x_hat[1] .+ x_hat[2] .* initial_x
    plot!(p2, initial_x, initial_line, label="Initial (n=$n)", linewidth=2, linestyle=:dash, markerstrokewidth=0)

    for iter in n+1:length(x)
        x_new = [1; x[iter]]  # Column vector
        y_new = y[iter]
        
        gain_vector = covar * x_new
        c = 1 / (1 + (x_new' * gain_vector))
        covar = covar - c * gain_vector * x_new' * covar
        error = y_new - (x_new' * x_hat)
        
        x_hat = x_hat + c * error * gain_vector
        
        # Plot this iteration's line with transparency based on iteration
        alpha = 0.4 + 0.6 * (iter - n) / (len - n)  # Increasing opacity
        iteration_line = x_hat[1] .+ x_hat[2] .* initial_x
        if iter % 2 == 0  # Plot every two iterations
            plot!(p2, initial_x, iteration_line, 
                  label="After point $iter", 
                  linewidth=1.5,
                  alpha=alpha)
        end
    end

    #y = b + mx
    final_line = x_hat[1] .+ ( x_hat[2] .* initial_x)
    plot!(p2, initial_x, final_line, label="Final model", linewidth=3, color=:red)
    
    return covar, x_hat, p1, p2
end

function CovarRecurrentRegression_plot(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Call computation function
    covar, x_hat = CovarRecurrentRegression_compute(x, y, n)
    
    println("Kalman Filter Coefficients: Intercept: ", x_hat[1], ", Slope: ", x_hat[2])
    
    # Plotting
    p = scatter(x, y, label="Data", title="Kalman Filter Regression", 
                markersize=5, color=:blue, markerstrokewidth=0)
    plot!(p, x, x_hat[1] .+ x_hat[2] .* x, label="Kalman Filter Regression", 
          color=:green, linewidth=2)
          
    return covar, x_hat, p
end

function compareRegressions(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Get your recursive regression result
    recursive_covar, recursive_x_hat, p1, p2 = recursiveRegression_plot(x, y, n)
    
    # Built-in regression using all data
    X = hcat(ones(length(x)), x)
    builtin_x_hat = X \ y
    
    # GLM regression
    df = DataFrame(X=x, Y=y)
    glm_model = lm(@formula(Y ~ X), df)
    glm_x_hat = coef(glm_model)
    
    # Compare results
    println("\nRegression Coefficients Comparison:")
    println("Method     | Intercept  | Slope")
    println("-----------|------------|------------")
    println("Recursive  | $(round(recursive_x_hat[1], digits=4)) | $(round(recursive_x_hat[2], digits=4))")
    println("Built-in   | $(round(builtin_x_hat[1], digits=4)) | $(round(builtin_x_hat[2], digits=4))")
    println("GLM        | $(round(glm_x_hat[1], digits=4)) | $(round(glm_x_hat[2], digits=4))")
    
    # Create a new plot for comparison
    p3 = plot(title="Regression Comparison")

    # Training points (first n points) in blue
    scatter!(p3, x[1:n], y[1:n], label="Training Data (1-$n)", 
             markersize=5, color=:blue, markerstrokewidth=0)
    
    # New points (n+1 to end) in orange
    scatter!(p3, x[n+1:end], y[n+1:end], label="New Data ($(n+1)-$(length(x)))", 
             markersize=5, color=:orange, markershape=:diamond, markerstrokewidth=0)
    
    # Plot each regression line
    plot_x = range(minimum(x), maximum(x), length=100)
    plot!(p3, plot_x, recursive_x_hat[1] .+ recursive_x_hat[2] .* plot_x, 
          label="Recursive", linewidth=2)
    plot!(p3, plot_x, builtin_x_hat[1] .+ builtin_x_hat[2] .* plot_x, 
          label="Built-in", linewidth=2, linestyle=:dash)
    plot!(p3, plot_x, glm_x_hat[1] .+ glm_x_hat[2] .* plot_x, 
          label="GLM", linewidth=2, linestyle=:dot)
    
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(1500, 1500))
    display(final_plot)
    
    return final_plot
end

function plotAllRegressions(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # 1. Standard Linear Regression (first n points)
    _, x_hat, p1 = linearRegression_plot(x, y, n)
    title!(p1, "Linear Regression (first $n points)")
    
    # 2. Recursive Least Squares
    _, recursive_x_hat, _, p2 = recursiveRegression_plot(x, y, n)
    title!(p2, "Recursive Least Squares Evolution")
    
    # 3. Covariance-based Kalman Filter
    _, kalman_x_hat, p3 = CovarRecurrentRegression_plot(x, y, n)
    title!(p3, "Kalman Filter Regression")
    
    # Print comparison of coefficients
    println("\n=== Regression Coefficients Comparison ===")
    println("Method     | Intercept  | Slope      | Description")
    println("-----------|------------|------------|------------")
    println("Linear     | $(round(x_hat[1], digits=4)) | $(round(x_hat[2], digits=4)) | Using first $n points")
    println("Recursive  | $(round(recursive_x_hat[1], digits=4)) | $(round(recursive_x_hat[2], digits=4)) | Sequential updates")
    println("Kalman     | $(round(kalman_x_hat[1], digits=4)) | $(round(kalman_x_hat[2], digits=4)) | Covariance-based")
    
    # Create comparison plot with all three methods
    p4 = plot(title="Comparison of All Methods", legend=:outertopright)
    
    # Plot data points
    scatter!(p4, x, y, label="All Data Points", markersize=5, color=:black, alpha=0.6, markerstrokewidth=0)
    
    # Plot each method's regression line
    plot_x = range(minimum(x), maximum(x), length=100)
    plot!(p4, plot_x, x_hat[1] .+ x_hat[2] .* plot_x, 
          label="Linear (n=$n)", linewidth=2, color=:blue)
    plot!(p4, plot_x, recursive_x_hat[1] .+ recursive_x_hat[2] .* plot_x, 
          label="Recursive", linewidth=2, color=:red)
    plot!(p4, plot_x, kalman_x_hat[1] .+ kalman_x_hat[2] .* plot_x, 
          label="Kalman", linewidth=2, color=:green, linestyle=:dash)
    
    # Calculate metrics for each method
    linear_mse = mean((y .- (x_hat[1] .+ x_hat[2] .* x)).^2)
    recursive_mse = mean((y .- (recursive_x_hat[1] .+ recursive_x_hat[2] .* x)).^2)
    kalman_mse = mean((y .- (kalman_x_hat[1] .+ kalman_x_hat[2] .* x)).^2)
    
    # Print MSE comparison
    println("\n=== Mean Squared Error Comparison ===")
    println("Linear:    $(round(linear_mse, digits=4))")
    println("Recursive: $(round(recursive_mse, digits=4))")
    println("Kalman:    $(round(kalman_mse, digits=4))")
    
    # Create the final layout with all plots
    final_plot = plot(p1, p2, p3, p4, 
                      layout=(2,2), 
                      size=(1500, 1500),
                      margin=5Plots.mm)
    
    return final_plot, x_hat, recursive_x_hat, kalman_x_hat
end

# Function to create an enhanced version with detailed metrics
function plotAllRegressionsDetailed(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Get results from base function
    plot_result, x_hat, recursive_x_hat, kalman_x_hat = plotAllRegressions(x, y, n)
    
    # Built-in regression using all data for comparison
    X = hcat(ones(length(x)), x)
    builtin_x_hat = X \ y
    
    # Calculate metrics for full dataset performance
    metrics_df = DataFrame(
        Method = ["Linear (n=$n)", "Recursive", "Kalman", "Full Dataset"],
        Intercept = [x_hat[1], recursive_x_hat[1], kalman_x_hat[1], builtin_x_hat[1]],
        Slope = [x_hat[2], recursive_x_hat[2], kalman_x_hat[2], builtin_x_hat[2]],
        MSE = [
            mean((y .- (x_hat[1] .+ x_hat[2] .* x)).^2),
            mean((y .- (recursive_x_hat[1] .+ recursive_x_hat[2] .* x)).^2),
            mean((y .- (kalman_x_hat[1] .+ kalman_x_hat[2] .* x)).^2),
            mean((y .- (builtin_x_hat[1] .+ builtin_x_hat[2] .* x)).^2)
        ]
    )
    
    # Print the metrics in a nice table format
    println("\n=== Detailed Regression Analysis ===")
    println("Method            | Intercept  | Slope      | MSE")
    println("------------------|------------|------------|------------")
    for r in eachrow(metrics_df)
        println("$(lpad(r.Method, 18)) | $(rpad(round(r.Intercept, digits=4), 10)) | $(rpad(round(r.Slope, digits=4), 10)) | $(round(r.MSE, digits=6))")
    end
    
    return plot_result
end

# Example usage:
result = plotAllRegressionsDetailed(x, y, 10)
display(result)

function compareRegressions(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Get your recursive regression result
    recursive_covar, recursive_x_hat, p1, p2 = recursiveRegression_plot(x, y, n)
    # Built-in regression using all data
    X = hcat(ones(length(x)), x)
    builtin_x_hat = X \ y
    
    # GLM regression
    df = DataFrame(X=x, Y=y)
    glm_model = lm(@formula(Y ~ X), df)
    glm_x_hat = coef(glm_model)
    
    # Compare results
    println("\nRegression Coefficients Comparison:")
    println("Method     | Intercept  | Slope")
    println("-----------|------------|------------")
    println("Recursive  | $(round(recursive_x_hat[1], digits=4)) | $(round(recursive_x_hat[2], digits=4))")
    println("Built-in   | $(round(builtin_x_hat[1], digits=4)) | $(round(builtin_x_hat[2], digits=4))")
    println("GLM        | $(round(glm_x_hat[1], digits=4)) | $(round(glm_x_hat[2], digits=4))")
    
    # Create a new plot for comparison
    p3 = plot(title="Regression Comparison")

    # Training points (first n points) in blue
    scatter!(p3, x[1:n], y[1:n], label="Training Data (1-$n)", 
             markersize=5, color=:blue, markerstrokewidth=0)
    
    # New points (n+1 to end) in orange
    scatter!(p3, x[n+1:end], y[n+1:end], label="New Data ($(n+1)-$(length(x)))", 
             markersize=5, color=:orange, markershape=:diamond, markerstrokewidth=0)
    
    # Plot each regression line
    plot_x = range(minimum(x), maximum(x), length=100)
    plot!(p3, plot_x, recursive_x_hat[1] .+ recursive_x_hat[2] .* plot_x, 
          label="Recursive", linewidth=2)
    plot!(p3, plot_x, builtin_x_hat[1] .+ builtin_x_hat[2] .* plot_x, 
          label="Built-in", linewidth=2, linestyle=:dash)
    plot!(p3, plot_x, glm_x_hat[1] .+ glm_x_hat[2] .* plot_x, 
          label="GLM", linewidth=2, linestyle=:dot)
    
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(1500, 1500))
    display(final_plot)
end

# compareRegressions(x, y, 10)