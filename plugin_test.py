# python3 ./main.py ./Seq2Seq/Config/Machine/zuse_json.conf ./Seq2Seq/Config/Output/output.conf ./Seq2Seq/Config/Input/exp01.conf --experiment_path ./Seq2Seq/test_json3

import os, sys


def respond(text):
    sys.stdout.write(text + "\n$EOS$\n")
    sys.stdout.flush()

def loop():
    line = sys.stdin.readline().strip()
    while line != "kill":
        if line == "<GO>":
            #string = sys.stdin.readline().strip()
            #respond(string)
            print("giving prebuilt string to nn model")
            string = '{"!":"com.github.javaparser.ast.body.MethodDeclaration","body":{"!":"com.github.javaparser.ast.stmt.BlockStmt","statements":[{"!":"com.github.javaparser.ast.stmt.ExpressionStmt","expression":{"!":"com.github.javaparser.ast.expr.VariableDeclarationExpr","annotations":[],"modifiers":[],"variables":[{"!":"com.github.javaparser.ast.body.VariableDeclarator","initializer":{"!":"com.github.javaparser.ast.expr.IntegerLiteralExpr","value":"0"},"name":{"!":"com.github.javaparser.ast.expr.SimpleName","identifier":"i"},"type":{"!":"com.github.javaparser.ast.type.PrimitiveType","type":"INT","annotations":[]}}]}},{<INV>:<EMPTY>},{"!":"com.github.javaparser.ast.stmt.ReturnStmt","expression":{"!":"com.github.javaparser.ast.expr.IntegerLiteralExpr","value":"0"}}]},"type":{"!":"com.github.javaparser.ast.type.PrimitiveType","type":"INT","annotations":[]},"modifiers":["PRIVATE"],"name":{"!":"com.github.javaparser.ast.expr.SimpleName","identifier":"test"},"parameters":[],"thrownExceptions":[],"typeParameters":[],"annotations":[]}'

            #string = "{ ' ! ' : ' com.github.javaparser.ast.body.MethodDeclaration ' , ' thrownExceptions ' : [ ] , ' type ' : { ' ! ' : ' com.github.javaparser.ast.type.VoidType ' , ' annotations ' : [ ] } , ' parameters ' : [ { ' isVarArgs ' : ' false ' , ' ! ' : ' com.github.javaparser.ast.body.Parameter ' , ' type ' : { ' ! ' : ' com.github.javaparser.ast.type.PrimitiveType ' , ' annotations ' : [ ] , ' type ' : ' INT ' } , ' name ' : { ' ! ' : ' com.github.javaparser.ast.expr.SimpleName ' , ' identifier ' : ' spacing ' } , ' annotations ' : [ ] , ' modifiers ' : [ ' FINAL ' ] , ' varArgsAnnotations ' : [ ] } ] , ' name ' : { ' ! ' : ' com.github.javaparser.ast.expr.SimpleName ' , ' identifier ' : ' setSpacing ' } , ' annotations ' : [ ] , ' modifiers ' : [ ' PUBLIC ' ] , ' typeParameters ' : [ ] , ' body ' : { ' ! ' : ' com.github.javaparser.ast.stmt.BlockStmt ' , ' statements ' : [ { ' ! ' : ' com.github.javaparser.ast.stmt.ExpressionStmt ' , ' expression ' : { ' value ' : { ' name ' : { ' ! ' : ' com.github.javaparser.ast.expr.SimpleName ' , ' identifier ' : ' spacing ' } , ' ! ' : ' com.github.javaparser.ast.expr.NameExpr ' } , ' ! ' : ' com.github.javaparser.ast.expr.AssignExpr ' , ' target ' : { ' name ' : { ' ! ' : ' com.github.javaparser.ast.expr.SimpleName ' , ' identifier ' : ' spacing ' } , ' ! ' : ' com.github.javaparser.ast.expr.FieldAccessExpr ' , ' scope ' : { ' ! ' : ' com.github.javaparser.ast.expr.ThisExpr ' } } , ' operator ' : ' ASSIGN ' } } , { ' ! ' : ' com.github.javaparser.ast.stmt.ExpressionStmt ' , ' expression ' : { ' name ' : { ' ! ' : ' com.github.javaparser.ast.expr.SimpleName ' , ' identifier ' : ' updateLayout ' } , ' ! ' : ' com.github.javaparser.ast.expr.MethodCallExpr ' , ' arguments ' : [ { ' name ' : { ' ! ' : ' com.github.javaparser.ast.expr.SimpleName ' , ' identifier ' : ' toolBar ' } , ' ! ' : ' com.github.javaparser.ast.expr.NameExpr ' } ] } } ] } }"
            os.system(
                'python3 ./main.py ./Seq2Seq/Config/Machine/zuse_json.conf ./Seq2Seq/Config/Output/output.conf ./Seq2Seq/Config/Input/exp01.conf --experiment_path ./Seq2Seq/test_json3 --input "' + string + '"')
        line = sys.stdin.readline().strip()

if __name__ == "__main__":
    loop()


