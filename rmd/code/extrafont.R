suppressPackageStartupMessages(library("extrafont")) # listado de fuentes del sistema
# font_import() # extrafont
# fonts() # extrafont
# fonttable() # extrafont

fuentes <- fonts() # extrafont

for (i in 1:length(fonts())) {
  for (j in 0:9) {
    filename <- paste("/numeros/", j, "/", fuentes[i], ".png", sep = "")
    png(filename = filename, width = 30, height = 30)
    par(mfrow = c(1, 1), mar = c(0,0,0,0))
    plot(-1, -1, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE)
    text(x = 0.5, y = 0.5, family = fuentes[i], j, cex = 3)    
    dev.off()
  }
}
