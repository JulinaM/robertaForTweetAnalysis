change conda env to gpu2
export FLASK_APP=app
export FLASK_ENV=development

flask run
curl -X POST -H "Content-Type: application/json" -d '{"tweet":"Some shots from my shoot for F.I.T. Studio. Coach Fred"}' http://localhost:5000/
