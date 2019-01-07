require('kernlab')

kfunction <- function(linear = 0, quadratic = 0)
{
  k <- function (x,y)
  {
    linear*sum((x)*(y)) + quadratic*sum((x^2)*(y^2))
  }
  class(k) <- "kernel"
  k
}

n <- 25
a1 <- rnorm(n)
a2 <- 1 - a1 + 2* runif(n)
b1 <- rnorm(n)
b2 <- -1 - b1 - 2*runif(n)
x <- rbind(matrix(cbind(a1, a2), ncol = 2), matrix(cbind(b1, b2), ncol = 2))
y <- matrix(c(rep(1, n), rep(-1, n)))

svp <- ksvm(x,
            y,
            type = "C-svc",
            C = 100, 
            kernel = kfunction(1, 0),
            scaled = c())

plot(range(x[, 1]),
     range(x[, 2]),
     type = 'n',
     xlab = expression(X[1]),
     ylab = expression(X[2]))
title(main = 'Caracteristicas separables lineales')

ymat <- ymatrix(svp)
points(x = x[-SVindex(svp), 1], 
       y = x[-SVindex(svp), 2], 
       pch = ifelse(ymat[-SVindex(svp)] < 0, 2, 1)) # 1: círculo 2: triángulo
points(x = x[SVindex(svp), 1], 
       y = x[SVindex(svp), 2], 
       pch = ifelse(ymat[SVindex(svp)] < 0, 17, 16)) # 16: círculo cerrado 17: triángulo cerrado

# Extraemos el vector w y b del modelo
w <- colSums(coef(svp)[[1]] * x[SVindex(svp), ])
b <- b(svp)

# Dibujamos las líneas
abline(b/w[2], -w[1]/w[2])
abline((b + 1)/w[2], -w[1]/w[2], col = "gray")
abline((b - 1)/w[2], -w[1]/w[2], col = "gray")
