# Impromptune

Read the paper here: https://scholars.unh.edu/honors/564/

Impromptune is a symbolic music generator using relative attention mechanisms. Given a short amount of music in a custom piano-roll-esque format, it uses this as a "prompt" and writes more music to continue the piece.

The main idea behind this model is a "Predictive Attention" layer: using a relative attention mechanism in a recurrent setting to continuously produce outputs on a sequence. With this layer at the end of a stack of Transformer-esque attention layers, the model is able to properly use the context of prior notes to continue the piece.
