# using Pkg
# Pkg.add("Plots")
# Pkg.add("DataFrames")
# Pkg.add("GLM")
using Plots # To graph points
using GLM, DataFrames # To calculate regression via julia's GLM package

points = rand(Float64, 16,2) * 10
x = points[:,1]
y = points[:,2]

function linearRegression(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Limit to n points
    x = x[1:n]
    A = hcat(ones(n), x)
    y = y[1:n]
    A_inv = inv(A' * A)
    x_hat = A_inv * (A' * y)       
    #x_hat = A \ y  # Alternative way to calculate x_hat 
    # Its faster since it solves the system of equations directly instead of calculating the inverse
    # However, you can't save the inverse correlation matrix for recursive least squares                                                                                                                     
    ŷ = A * x_hat
    
    println("Intercept: ", x_hat[1], ", Slope: ", x_hat[2])
    
    # Create a plot but don't display it yet
    p = scatter(x, y, label="Data", title="Linear Regression (first $n points)", markersize=5, color=:blue,markerstrokewidth=0)
    plot!(p, x, ŷ, label="Regression", color=:green, linewidth=2)
    
    return A_inv, x_hat, p  # Return the plot too
end

function recursiveRegression(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Initial regression with first n points
    covar, x_hat, p1 = linearRegression(x, y, n)

    # Create second plot for showing recursive updates
    len = length(x)
    # Create plot with different colored points
    p2 = plot(title="Recursive Least Squares Evolution", legend=:outertopright)
    
    # Training points (first n points) in blue
    scatter!(p2, x[1:n], y[1:n], label="Training Data (1-$n)", 
             markersize=5, color=:blue,markerstrokewidth=0)
    
    # New points (n+1 to end) in orange
    scatter!(p2, x[n+1:end], y[n+1:end], label="New Data ($(n+1)-$len)", 
             markersize=5, color=:orange, markershape=:diamond,markerstrokewidth=0)

    # Plot initial model
    initial_x = range(minimum(x), maximum(x), length=100)
    initial_line = x_hat[1] .+ x_hat[2] .* initial_x
    plot!(p2, initial_x, initial_line, label="Initial (n=$n)", linewidth=2, linestyle=:dash, markerstrokewidth=0)
    
    # Process remaining points recursively
    for iter in n+1:length(x)
        x_new = [1; x[iter]]  # Column vector
        y_new = y[iter]
        
        # Calculate gain vector
        gain_vector = covar * x_new
        
        # Calculate scalar c
        c = 1 / (1 + (x_new' * gain_vector))
        
        # Update inverse correlation matrix
        covar = covar - c * gain_vector * x_new' * covar
        
        # Calculate prediction error
        error = y_new - (x_new' * x_hat)
        
        # Update parameter vector
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
    
    # Plot final model with distinctive style
    #y = b + mx
    final_line = x_hat[1] .+ ( x_hat[2] .* initial_x)
    plot!(p2, initial_x, final_line, label="Final model", linewidth=3, color=:red)
    
    # Arrange plots in a layout
    # final_plot = plot(p1, p2, layout=(2,1), size=(1500, 1500))
    # display(final_plot)
    
    return covar, x_hat, p1, p2
end

# Call the function and capture the return values
covar, x_hat = recursiveRegression(x, y, 10)

function compareRegressions(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Get your recursive regression result
    recursive_covar, recursive_x_hat, p1, p2 = recursiveRegression(x, y, n)
    
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

# Run the comparison
compareRegressions(x, y, 10)
