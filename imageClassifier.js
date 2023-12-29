import * as tf from '@tensorflow/tfjs';


const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const { createCanvas, loadImage } = require('canvas');
const { promisify } = require('util');

const loadModel = async () => {
    return await tf.loadLayersModel('file://models/hab_detect/model.json');
};

const classifyImage = async (model, imagePath) => {
    const img = await loadImage(imagePath);
    const canvas = createCanvas(256, 256);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 256, 256);

    const resized = tf.browser.fromPixels(canvas).toFloat().div(255).expandDims();
    const prediction = model.predict(resized);
    const predictVal = prediction.dataSync()[0];

    if (predictVal > 0.5) {
        console.log(`Predicted class likely has an algae bloom ${imagePath}`);
        fs.appendFileSync('pab.txt', `${imagePath} ${predictVal}\n`);
    } else {
        console.log(`Predicted class likely does not have an algae bloom ${imagePath}`);
        fs.appendFileSync('npab.txt', `${imagePath} ${predictVal}\n`);
    }
};

const main = async () => {
    const sourceFolder = 'D:/DCIM/100MEDIA';
    const destinationFolder = 'C:/Users/25milanbm/Desktop/HAB_Detect/drone_data';

    if (fs.existsSync(sourceFolder)) {
        console.log(sourceFolder);

        const files = fs.readdirSync(sourceFolder);

        for (const fileName of files) {
            const source = `${sourceFolder}/${fileName}`;
            if (fs.statSync(source).isFile()) {
                const destination = `${destinationFolder}/${fileName}`;
                fs.copyFileSync(source, destination);
            }
        }

        const model = await loadModel();

        const dataDir = 'C:/Users/25milanbm/Desktop/HAB_Detect/drone_data';
        const images = fs.readdirSync(dataDir);

        for (const image of images) {
            const imagePath = `${dataDir}/${image}`;
            await classifyImage(model, imagePath);
        }
    }
};

main();