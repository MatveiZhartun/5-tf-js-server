const fs = require('fs');
const CSV = require('comma-separated-values');
const trainTestSplit = require('train-test-split');
const ConfusionMatrix = require('ml-confusion-matrix');
const _ = require('lodash');
const tf = require('@tensorflow/tfjs-node');
const wait = () => setTimeout(wait, 1000);
wait();

SLICE_INDEX = 100

function roll(v) {
  return _.indexOf(v, _.max(v));
}

function unroll(v) {
  let t = _.fill(Array(10), 0);
  
  t[v] = 1;

  return t;
}

let allData = CSV.parse(fs.readFileSync(('./data/train.csv'), 'utf-8')).slice(1, SLICE_INDEX)
let [trainData, validationData] = trainTestSplit(allData, 0.8, 1234)

let yTrainData = _.map(trainData, (d) => Number(d[0]))
let xTrainData = _.map(trainData, (d) => _.slice(d, 1))

let yValidationData = _.map(validationData, (d) => Number(d[0]))
let xValidationData = _.map(validationData, (d) => _.slice(d, 1))

xTrainData = tf.tensor2d(_.map(xTrainData, (row) => _.map(row, (value) => value / 255)))
xValidationData = tf.tensor2d(_.map(xValidationData, (row) => _.map(row, (value) => value / 255)))

yTrainData = tf.tensor2d(_.map(yTrainData, (v) => unroll(v)))

let model = tf.sequential()
model.add(tf.layers.dense({ units: 1000, inputShape: 784, activation: 'relu' }))
model.add(tf.layers.dense({ units: 500, activation: 'relu' }))
model.add(tf.layers.dense({ units: 100, activation: 'relu' }))
model.add(tf.layers.dense({ units: 10, activation: 'sigmoid' }))
model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] })

const onBatchEnd = (__, logs) => console.log('[Batch]', logs);
const onEpochEnd = (__, logs) => console.log('[Epoch]', logs);

model.fit(xTrainData, yTrainData, { epochs: 1, batchSize: 100, validationSplit: 0.2, callbacks: {onBatchEnd, onEpochEnd} })
  .then(() => {
    let yResult = model.predict(xValidationData).arraySync();
    let CM2 = ConfusionMatrix.fromLabels(yValidationData, _.map(yResult, (row) => roll(row)));

    console.log('Accuracy: ' + CM2.getAccuracy());
  })
  .then(() => {
    model.save('file://./model').catch((e) => console.log(e));
  })
  .catch((e) => console.log(e));


