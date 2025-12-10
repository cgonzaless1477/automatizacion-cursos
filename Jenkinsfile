pipeline {
    agent any

    triggers {
        pollSCM('H/2 * * * *')
    }

    stages {
        stage('Build & Test') {
            steps {
                dir('curso-automatizacion-web') {   // 👈 MUY IMPORTANTE
                    bat 'mvn clean test'
                }
            }
        }
    }

    post {
        always {
            dir('curso-automatizacion-web') {
                junit 'target/surefire-reports/*.xml'

                archiveArtifacts artifacts: 'target/screenshots/**/*.png',
                                 fingerprint: true,
                                 allowEmptyArchive: true
            }
        }
    }
}
