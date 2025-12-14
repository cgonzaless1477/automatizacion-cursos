pipeline {
    agent any

    environment {
        JMETER_HOME        = 'C:\\apache-jmeter-5.6.3'
        JMETER_TEST_PLAN   = 'test\\escenario-principal.jmx'
        JMETER_RESULTS_DIR = 'target\\jmeter'

        JMETER_JTL        = 'target\\jmeter\\resultados.jtl'
        JMETER_REPORT_DIR = 'target\\jmeter\\report'

        CUSTOM_HTML = 'target\\performance_report_custom.html'
        CUSTOM_PDF  = 'target\\performance_report_custom.pdf'

        PYTHON_EXE = 'C:\\Users\\intel\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'
    }

    stages {

        stage('Verificar entorno') {
            steps {
                dir('curso-performance-jmeter') {
                    bat '''
                        echo ===== Verificando JMeter =====
                        "%JMETER_HOME%\\bin\\jmeter.bat" -v
                    '''
                }
            }
        }

        stage('Ejecutar pruebas de performance') {
            steps {
                dir('curso-performance-jmeter') {
                    bat '''
                        echo ===== Carpeta actual =====
                        cd

                        REM Crear carpeta base de resultados si no existe
                        if not exist "%JMETER_RESULTS_DIR%" mkdir "%JMETER_RESULTS_DIR%"

                        REM Borrar reporte anterior si existe
                        if exist "%JMETER_REPORT_DIR%" rmdir /s /q "%JMETER_REPORT_DIR%"

                        REM Borrar archivo de resultados anterior si existe
                        if exist "%JMETER_JTL%" del /f /q "%JMETER_JTL%"

                        REM Validar que exista el test plan
                        if not exist "%JMETER_TEST_PLAN%" (
                          echo ERROR: No existe el JMX: %JMETER_TEST_PLAN%
                          exit /b 1
                        )

                        echo ===== Ejecutando JMeter (non-GUI) =====
                        "%JMETER_HOME%\\bin\\jmeter.bat" -n ^
                          -t "%JMETER_TEST_PLAN%" ^
                          -l "%JMETER_JTL%" ^
                          -e -o "%JMETER_REPORT_DIR%"

                        echo ===== Codigo de salida JMeter: %ERRORLEVEL% =====
                        exit /b %ERRORLEVEL%
                    '''
                }
            }
        }

        stage('Generar reporte HTML y PDF personalizado') {
            steps {
                dir('curso-performance-jmeter') {
                    bat '''
                        setlocal EnableExtensions
                        echo ===== Generando reporte custom =====

                        "%PYTHON_EXE%" --version
                        if %ERRORLEVEL% NEQ 0 (
                          echo ERROR: No se pudo ejecutar Python en %PYTHON_EXE%
                          exit /b 1
                        )

                        "%PYTHON_EXE%" -m pip --version
                        if %ERRORLEVEL% NEQ 0 (
                          echo ERROR: pip no esta disponible
                          exit /b 1
                        )

                        REM deps (incluye matplotlib)
                        "%PYTHON_EXE%" -m pip install --user --upgrade --no-warn-script-location pandas numpy reportlab matplotlib
                        if %ERRORLEVEL% NEQ 0 (
                          echo ERROR: Fallo instalando dependencias Python
                          exit /b 1
                        )

                        if not exist "scripts\\generate_custom_report.py" (
                          echo ERROR: No existe scripts\\generate_custom_report.py
                          exit /b 1
                        )

                        "%PYTHON_EXE%" scripts\\generate_custom_report.py --jtl "%JMETER_JTL%" --out "target" ^
  --job "%JOB_NAME%" --build "%BUILD_NUMBER%" --url "%BUILD_URL%" ^
  --sla_p95 800 --sla_err 1

                        if %ERRORLEVEL% NEQ 0 (
                          echo ERROR: Fallo el script Python
                          exit /b 1
                        )

                        if not exist "%CUSTOM_HTML%" (
                          echo ERROR: No se genero el HTML custom: %CUSTOM_HTML%
                          exit /b 1
                        )

                        if not exist "%CUSTOM_PDF%" (
                          echo ERROR: No se genero el PDF custom: %CUSTOM_PDF%
                          exit /b 1
                        )

                        echo OK: Reporte custom generado
                    '''
                }
            }
        }
    }

    post {
        always {
            dir('curso-performance-jmeter') {

                // Publicar HTML custom (debe ir en post/steps, NO suelto)
                publishHTML(target: [
                    reportDir: 'target',
                    reportFiles: 'performance_report_custom.html',
                    reportName: 'Performance Custom Report',
                    keepAll: true,
                    alwaysLinkToLastBuild: true,
                    allowMissing: false
                ])

                // Guardar artefactos: JMeter + custom
                archiveArtifacts artifacts: 'target/**/*',
                                 fingerprint: true,
                                 allowEmptyArchive: true

                // Plugin Performance
                perfReport(
                    errorFailedThreshold: 10,
                    errorUnstableThreshold: 1,
                    sourceDataFiles: 'target/jmeter/resultados.jtl'
                )
            }
        }
    }
}
