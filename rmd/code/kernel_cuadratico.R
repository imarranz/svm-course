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

n <- 20
r <- runif(n)
a <- 2*pi*runif(n)
a1 <- r*sin(a)
a2 <- r*cos(a)
r <- 2 + runif(n)
a <- 2*pi*runif(n)
b1 <- r*sin(a)
b2 <- r*cos(a)
x <- rbind(matrix(cbind(a1, a2), ncol = 2), matrix(cbind(b1, b2), ncol = 2))
y <- matrix(c(rep(1, n), rep(-1, n)))

svp <- ksvm(x,
            y,
            type = "C-svc",
            C = 100, 
            kernel = kfunction(0, 1),
            scaled = c())

par(mfrow = c(1, 2))
plot(range(x[, 1]),
     range(x[, 2]),
     type = 'n',
     xlab = expression(X[1]),
     ylab = expression(X[2]))

title(main = 'Espacio de caracteristicas')
ymat <- ymatrix(svp)
points(x = x[-SVindex(svp),1], 
       y = x[-SVindex(svp),2], 
       pch = ifelse(ymat[-SVindex(svp)] < 0, 2, 1))
points(x = x[SVindex(svp),1], 
       y = x[SVindex(svp),2], 
       pch = ifelse(ymat[SVindex(svp)] < 0, 17, 16))

# Extraemos el vector w y b del modelo
w2 <- colSums(coef(svp)[[1]] * x[SVindex(svp), ]^2)
b <- b(svp)

x1 <- seq(min(x[, 1]), max(x[, 1]), 0.01)
x2 <- seq(min(x[, 2]), max(x[, 2]), 0.01)

points(-sqrt((b-w2[1]*x2^2)/w2[2]), x2, pch = 16 , cex = .2 )
points(sqrt((b-w2[1]*x2^2)/w2[2]), x2, pch = 16 , cex = .2 )
points(x1, sqrt((b-w2[2]*x1^2)/w2[1]), pch = 16 , cex = .2 )
points(x1, -sqrt((b-w2[2]*x1^2)/w2[1]), pch = 16, cex = .2 )

points(-sqrt((1+ b-w2[1]*x2^2)/w2[2]) , x2, pch = 16 , cex = .2 , col = "gray")
points( sqrt((1 + b-w2[1]*x2^2)/w2[2]) , x2,  pch = 16 , cex = .2 , col = "gray")
points( x1 , sqrt(( 1 + b -w2[2]*x1^2)/w2[1]), pch = 16 , cex = .2 , col = "gray")
points( x1 , -sqrt(( 1 + b -w2[2]*x1^2)/w2[1]), pch = 16, cex = .2 , col = "gray")

points(-sqrt((-1+ b-w2[1]*x2^2)/w2[2]) , x2, pch = 16 , cex = .2 , col = "gray")
points( sqrt((-1 + b-w2[1]*x2^2)/w2[2]) , x2,  pch = 16 , cex = .2 , col = "gray")
points( x1 , sqrt(( -1 + b -w2[2]*x1^2)/w2[1]), pch = 16 , cex = .2 , col = "gray")
points( x1 , -sqrt(( -1 + b -w2[2]*x1^2)/w2[1]), pch = 16, cex = .2 , col = "gray")

xsq <- x^2
svp <- ksvm(x = xsq,
            y = y,
            type = "C-svc",
            C = 100, 
            kernel = kfunction(1, 0),
            scaled = c())

plot(x = range(xsq[, 1]),
     y = range(xsq[, 2]),
     type = 'n',
     xlab = expression(X[1]^2),
     ylab = expression(X[2]^2))

title(main='Espacio cuadratico')
ymat <- ymatrix(svp)
points(x = xsq[-SVindex(svp), 1], 
       y = xsq[-SVindex(svp), 2], 
       pch = ifelse(ymat[-SVindex(svp)] < 0, 2, 1))
points(x = xsq[SVindex(svp), 1], 
       y = xsq[SVindex(svp), 2], 
       pch = ifelse(ymat[SVindex(svp)] < 0, 17, 16))

# Extraemos el vector w y b del modelo
w <- colSums(coef(svp)[[1]] * xsq[SVindex(svp),])
b <- b(svp)

# Dibujamos las líneas
abline(b/w[2], -w[1]/w[2])
abline((b + 1)/w[2], -w[1]/w[2], col = "gray")
abline((b - 1)/w[2], -w[1]/w[2], col = "gray")

