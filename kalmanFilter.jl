using Plots # To graph linear regression

points = rand(Float64, 16,2) * 10
x = points[:,1]

# By adding a column of ones we somehow are implementing a y intercept and are not forcing the 
# linear regression to go through the origin
y = points[:,2]


function linearRegression(x::Vector{Float64}, y::Vector{Float64},n::Int)
    # ' operator means transpose
    # \ operator means left division
    # For matrices it essentially does a fast inverse
    # Limit to 10 points
    x=x[1:n]
    A = hcat(ones(n), x)
    y = y[1:n]
    A_inv = inv(A' * A)
    x_hat = A_inv * (A' * y)                                                                                                                            
    ŷ = A * x_hat
    #ŷ = x_hat[1] .+ x_hat[2] .* x
    println("Slope: ", x_hat[2])
    scatter(x, y, label="Data")
    plot!(x, ŷ, label="Regression")
    return A_inv, x_hat
end

#linearRegression(x, y, 10)


# RLS
function recursiveRegression(x::Vector{Float64}, y::Vector{Float64}, n::Int)


    covar,x_hat = linearRegression(x, y, n)

    for iter in n+1:length(x)
        x_new = [1; x[iter]]  # Make this a column vector with semicolon
        y_new = y[iter]

        # In the book this should be covar * x_new transpose
        # However, julia treats vector as columns by default
        # So essentially the transpose are inverted
        gain_vector = covar * x_new
        
        # Calculate scalar c
        c = 1 / (1 + (x_new' * gain_vector))
        #c = gain_vector / (1 + (x_new' * gain_vector)) #Alternative calculate

        # Update inverse correlation matrix
        covar = covar - c * gain_vector * x_new' * covar
        
        # Calculate prediction error
        error = y_new - (x_new' * x_hat)
        
        # Update parameter vector
        x_hat = x_hat + c * error * gain_vector
    end
    
    # Create design matrix for predictions
    A = hcat(ones(length(x)), x)
    # Make predictions
    ŷ = A * x_hat

    scatter(x, y, label="Data")
    plot!(x, ŷ, label="Updated Regression")
end

recursiveRegression(x, y, 10)

# function covariantRecurrentRegression(points::Vector{Float64})
    
# end