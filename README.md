<h2>LSTM for null cipher generation</h2>
This project is a Bidirectional LSTM for generating the next character in a list of strings. 

The idea is to build a system capable of producing human-like text. 
The current model maxes out its accuracy around 60%, which is quite decent, but still, nothing compared to a human.

The end goal is to be able to give the model a string you would like to have hidden, then have the program generate a human-sounding paragraph that 
hides your text using the famous first letter null cipher.

An example the model generated was as follows.

<b>Hidden text </b>  <i>i am the secret</i>

<b>Model produced text </b>  <i>street,  is  a  musking  the  hand  everything  staluling  equally  carry,"  reflect.  every  time in it of the 
consideration bulsed on that copulspose of wramur "to--stopped; now. is you laid to as he had made his words.</i>

If you read the first letter of each word, you will see it spells out our hidden text. However, you will also notice the text is a bunch of gibberish that makes
no sense and is spelled wrong about as often as it is correct.
This is because the model is only learning character-level patterns, and so most of the models' efforts are spent learning to spell words, not make coherent sentences.

A few ways to improve the results.

- Train a word level model
- Process the hidden text and try to reduce the length and frequency of letters that do not start words. 
- Use a known algorithm to encrypt the hidden text before hidding it in a paragraph.
