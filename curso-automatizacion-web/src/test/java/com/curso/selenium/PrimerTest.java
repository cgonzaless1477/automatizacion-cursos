package com.curso.selenium;

import org.junit.Test;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;

public class PrimerTest {

    @Test
    public void pruebaGoogle() throws IOException {
        ChromeOptions options = new ChromeOptions();
        // Si quieres, luego podemos dejarlo headless:
        // options.addArguments("--headless=new");

        WebDriver driver = new ChromeDriver(options);

        try {
            driver.get("https://www.google.com");

            System.out.println("Título de la página: " + driver.getTitle());

            // Aquí tomamos la captura de evidencia aaaabbbm
            takeScreenshot(driver, "google-home");
        } finally {
            driver.quit();
        }
    }

    private void takeScreenshot(WebDriver driver, String nombre) throws IOException {
        TakesScreenshot ts = (TakesScreenshot) driver;
        File src = ts.getScreenshotAs(OutputType.FILE);

        Path destinoDir = Paths.get("target", "screenshots");
        Files.createDirectories(destinoDir);

        String fileName = nombre + "-" + System.currentTimeMillis() + ".png";
        Path destino = destinoDir.resolve(fileName);

        Files.copy(src.toPath(), destino, StandardCopyOption.REPLACE_EXISTING);

        System.out.println("Screenshot guardado en: " + destino.toAbsolutePath());
    }
}
