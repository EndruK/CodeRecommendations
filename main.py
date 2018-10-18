import Seq2Seq.seq2seq as s2s
import Seq2Seq.generate as s2sGen
import Seq2SeqAtt.seq2seqAtt as s2sAtt
import Seq2SeqAtt.generate as s2sAttGen
import sys

# Mode: train | generate
MODE = "generate"
# Module: s2s | s2sAtt | s2sAttCopy | t2t"
RUN_MODULE = "s2s"

if __name__ == "__main__":
    if MODE == "train":
        if RUN_MODULE == "s2s":
            s2s.run()
        elif RUN_MODULE == "s2sAtt":
            s2sAtt.run()
        else:
            sys.exit(0)
    elif MODE == "generate":
        if RUN_MODULE == "s2s":
            s2sGen.seq2seq_generate()
        elif RUN_MODULE == "s2sAtt":
            s2sAttGen.run()
        else:
            sys.exit(0)
