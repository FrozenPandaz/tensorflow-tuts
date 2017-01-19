// const pythonShell = require('python-shell');
// const path = require('path');
file = './src/mnist/mnist.py';

const spawn = require('child_process').spawn;

proc = spawn('python3', [file])

proc.stdout.on('data', (str) => {
    console.log(`${str}`);
});

proc.stderr.on('data', (str) => {
    console.log(`${str}`);
});

proc.on('close', (code) => {
    console.log('process exited with', code);
});

// pythonShell.run(file, (err) => {
//     if (err) {
//         throw err
//     }
//     console.log('finished');
// });