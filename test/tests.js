var should = require('chai').should(),
    neuralnet = require('../index');

describe('#equals', function() {
  this.timeout(500000);

  it('verifies variable can be accessed', function() {
    neuralnet.credit().should.equal("kevin");
  });
  it('create hidden neuron and check forward functionality', function() {
    neuralnet.neuron().should.equal(3);
  });
  it('create hidden layer and check forward functionality', function() {
    neuralnet.layer().should.eql([3,3]);
  });
  it('create neural network and check forward functionality', function() {
    neuralnet.network().should.eql([15,15]);
  });
  it('gradient properly adjusts weights', function() {
    neuralnet.adjustment().should.be.below(neuralnet.xorStructure());
  });

  it('XOR with one hidden layer, 3 neurons', function() {
    var net = neuralnet.trainXOR();
    net.forward([0,1]).should.be.above(0.5);
    console.log(net.forward([0,1]));
    net.forward([1,1]).should.be.below(0.5);
    console.log(net.forward([1,1]));
    net.forward([0,0]).should.be.below(0.5);
    console.log(net.forward([0,0]));
    net.forward([1,0]).should.be.above(0.5);
    console.log(net.forward([1,0]));
  });

  it('XOR with 2 hiddens, 10 neurons', function() {
    var net2 = neuralnet.trainMultiXOR();
    net2.forward([0,1]).should.be.above(0.5);
    console.log(net2.forward([0,1]));
    net2.forward([1,1]).should.be.below(0.5);
    console.log(net2.forward([1,1]));
    net2.forward([0,0]).should.be.below(0.5);
    console.log(net2.forward([0,0]));
    net2.forward([1,0]).should.be.above(0.5);
    console.log(net2.forward([1,0]));
  });

  it('test network with multiple outputs', function() {
    var netRGB = neuralnet.trainMultipleOutput();
    console.log(netRGB.forward([1,0]));
    console.log(netRGB.forward([0,0]));
    console.log(netRGB.forward([1,1]));
    console.log(netRGB.forward([0,1]));
    netRGB.forward([1,0])[0].should.be.above(0.5); netRGB.forward([1,0])[1].should.be.below(0.5);
    netRGB.forward([0,0])[0].should.be.below(0.5); netRGB.forward([0,0])[1].should.be.below(0.5);
    netRGB.forward([1,1])[0].should.be.above(0.5); netRGB.forward([1,1])[1].should.be.above(0.5);
    netRGB.forward([0,1])[0].should.be.below(0.5); netRGB.forward([0,1])[1].should.be.above(0.5);
  });

  it('train network to switch the inputs', function() {
    var netRGB = neuralnet.trainOpposites();
    console.log(netRGB.forward([1,0]));
    console.log(netRGB.forward([0,0]));
    console.log(netRGB.forward([1,1]));
    console.log(netRGB.forward([0,1]));
    netRGB.forward([1,0])[0].should.be.below(0.5); netRGB.forward([1,0])[1].should.be.above(0.5);
    netRGB.forward([0,0])[0].should.be.below(0.5); netRGB.forward([0,0])[1].should.be.below(0.5);
    netRGB.forward([1,1])[0].should.be.above(0.5); netRGB.forward([1,1])[1].should.be.above(0.5);
    netRGB.forward([0,1])[0].should.be.above(0.5); netRGB.forward([0,1])[1].should.be.below(0.5);
  });

  it('test network to recognize Flat UI colors.', function() {

    var netRGB = neuralnet.trainColors();

    var red = netRGB.forward([14,7,4,12,3,12]);
    var blue = netRGB.forward([4,11,7,7,11,14]);
    var green = netRGB.forward([0,0,11,1,6,10]);
    var orange = netRGB.forward([15,8,9,4,0,6]);
    var purple = netRGB.forward([9,11,5,9,11,6]);

    console.log("red " + red + " " + chooseOne(red));
    console.log("blue " + blue + " " + chooseOne(blue));
    console.log("green " + green + " " + chooseOne(green));
    console.log("orange " + orange + " " + chooseOne(orange));
    console.log("purple " + purple + " " + chooseOne(purple));

    chooseOne(red).should.eql(0);
    chooseOne(blue).should.eql(1);
    chooseOne(green).should.eql(2);
    chooseOne(orange).should.eql(3);
    chooseOne(purple).should.eql(4);

  });
  // it('train to recognise handwriting', function() {
  //   var net = neuralnet.trainDigits();
  //   // net.forward([0,1]).should.be.above(0.5);
  //   // console.log(net.forward([0,1]));
  //   // net.forward([1,1]).should.be.below(0.5);
  //   // console.log(net.forward([1,1]));
  //   // net.forward([0,0]).should.be.below(0.5);
  //   // console.log(net.forward([0,0]));
  //   // net.forward([1,0]).should.be.above(0.5);
  //   // console.log(net.forward([1,0]));
  // });



});


































function chooseOne(results)
{
  var highest = 0;
  var highestindex = 0;
  for(var i = 0; i < results.length; i++)
  {
    if(results[i] > highest)
    {
      highest = results[i];
      highestindex = i;
    }
  }
  return highestindex;
}
