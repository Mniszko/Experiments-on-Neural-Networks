if ! test -f ./mnist_red/optdigits.tra; then
    wget https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip
    mkdir mnist_red
    unzip ./optical+recognition+of+handwritten+digits.zip -d mnist_red
fi
if ! test -f ./multiplexer_FNN_distances.csv; then
    touch ./multiplexer_FNN_distances.csv
    touch ./multiplexer_FNN_accuracies.csv
fi
