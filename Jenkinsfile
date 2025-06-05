pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "gen-lang-client-0422234397"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
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
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        stage('Building and Pushing Docker image to GCR'){
            steps{
                withCredentials([file(credentialsId: 'gcp-key', variable : 'GOOGLE_APP_CREDENTIALS')]){
                    script{
                        echo 'Building and Pushing Docker image to GCR.............'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APP_CREDENTIALS}

                        gcloud config set project ${GCP_PROJECT}

                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

                        docker push gcr.io/${GCP_PROJECT}/ml-project:latest
                        '''
                    }
                }
                
            }
        }

        stage('Deployed to Google Cloud Run'){
            steps{
                withCredentials([file(credentialsId: 'gcp-key', variable : 'GOOGLE_APP_CREDENTIALS')]){
                    script{
                        echo 'Deployed to Google Cloud Run............'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APP_CREDENTIALS}

                        gcloud config set project ${GCP_PROJECT}

                        gcloud run deploy ml-project \
                        --platform=managed \
                            --image=gcr.io/${GCP_PROJECT}/ml-project:latest \
                            --region=us-central1 \
                            --allow-unauthenticated
                        '''
                    }
                }
                
            }
        }
    }
}