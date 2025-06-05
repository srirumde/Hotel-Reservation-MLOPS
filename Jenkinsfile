pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins....'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/srirumde/Hotel-Reservation-MLOPS.git']])
                }
            }
        }

        stage('Setting Up our Virtual Env and Installing Dependencies'){
            steps{
                script{
                    echo 'Setting Up our Virtual Env and Installing Dependencies...........'
                    sh '''
                    python -m venv ${VENV_DIR}
                    .${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
    }
}