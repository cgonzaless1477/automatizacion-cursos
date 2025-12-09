pipeline {
    agent any

    // 🔔 Disparador automático: revisa el repo cada 2 minutos
    triggers {
        pollSCM('H/2 * * * *')
        // Ejemplos:
        // H/5 * * * *  -> cada 5 minutos
        // H * * * *    -> una vez por hora
    }

    stages {
        stage('Build & Test') {
            steps {
                bat 'mvn clean test'
            }
        }
    }

    post {
        always {
            // Publicar resultados de tests
            junit 'target/surefire-reports/*.xml'

            // Publicar capturas como artefactos
            archiveArtifacts artifacts: 'target/screenshots/**/*.png',
                             fingerprint: true,
                             allowEmptyArchive: true
        }
    }
}
