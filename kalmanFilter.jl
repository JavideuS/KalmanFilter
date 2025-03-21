using Plots # To graph linear regression

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
    ŷ = A * x_hat
    
    println("Intercept: ", x_hat[1], ", Slope: ", x_hat[2])
    
    # Create a plot but don't display it yet
    p = scatter(x, y, label="Data", title="Linear Regression (first $n points)", markersize=5)
    plot!(p, x, ŷ, label="Regression", color=:green, linewidth=2)
    
    return A_inv, x_hat, p  # Return the plot too
end

function recursiveRegression(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    # Initial regression with first n points
    covar, x_hat, p1 = linearRegression(x, y, n)

    # Process remaining points recursively
    for iter in n+1:length(x)
        x_new = [1; x[iter]]  # Make this a column vector with semicolon
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
        
    end
    
    # Create a second plot for the recursive regression
    len = length(x)
    p2 = scatter(x, y, label="Data", markersize=5, title="Recursive Least Squares $len points ")
    
    # Plot the final model applied to all data
    A = hcat(ones(length(x)), x)
    ŷ = A * x_hat
    plot!(p2, x, ŷ, label="Complete Regression", linewidth=2, color=:darkorange)
    
    # Arrange both plots side by side
    final_plot = plot(p1, p2, layout=(1,2), size=(1000, 400))
    display(final_plot)  # Explicitly display the combined plot
    
    return covar, x_hat
end

# Call the function and capture the return values
covar, x_hat = recursiveRegression(x, y, 10)
