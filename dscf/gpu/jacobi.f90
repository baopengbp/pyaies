      SUBROUTINE diag_a(A,U,E,NDIM)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
C  ....  THIS ROUTINE DIAGONALIZES THE INPUT MATRIX, A, BY THE
C  ....  JACOBI METHOD, PLACING THE EIGENVECTORS IN THE COLUMNS OF
C  ....  U, AND THE EIGENVALUES IN THE DIAGONAL OF A.
C********************************************************************C
C   ARGUMENTS.                                                       C
C        A       MATRIX TO BE DIAGONALIZED.                          C
C        U       MATRIX OF EIGENVECTORS.                             C
C        E       LIST OF EIGENVALUES                                 C
C        N       DIMENSION OF THE MATRICES AND THE VECTOR.           C
C********************************************************************C
      DOUBLE PRECISION, INTENT(INOUT) :: A(NDIM,NDIM),U(NDIM,NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: E(NDIM)
      DATA ZER,ONE,TWO,FOR,EPS/0.0D 00,1.0D 00,2.0D 00,4.0D 00,1.0D-20/

      N=NDIM

      DO 8 J=1,N
      DO 9 I=1,N
    9 U(I,J)=ZER
    8 U(J,J)=ONE
   10 AMAX=ZER
      DO 11 I=2,N
      JUP=I-1
      DO 11 J=1,JUP
      AII=A(I,I)
      AJJ=A(J,J)
      AOD=A(I,J)
      ASQ=AOD*AOD
   28 IF(ASQ-AMAX) 23,23,27
   27 AMAX=ASQ
   23 IF(ASQ-EPS)  11,11,12
   12 DIFFR=AII-AJJ
      IF(DIFFR)  13,15,15
   13 SIGN=-TWO
      DIFFR=-DIFFR
      GO TO 16
   15 SIGN=TWO
   16 TDEN=DIFFR+ DSQRT(DIFFR*DIFFR+FOR*ASQ)
      TAN=SIGN*AOD/TDEN
      C=ONE/( DSQRT(ONE+TAN*TAN))
      S=C*TAN
      DO 24 K=1,N
      XJ=C*U(K,J)-S*U(K,I)
      U(K,I)=S*U(K,J)+C*U(K,I)
      U(K,J)=XJ
      IF(K-J)  18,24,18
   18 IF(K-I)  21,24,21
   21 XJ=C*A(K,J)-S*A(K,I)
      A(K,I)=S*A(K,J)+C*A(K,I)
      A(K,J)=XJ
      A(I,K)=A(K,I)
      A(J,K)=A(K,J)
   24 CONTINUE
      A(I,I)=C*C*AII+S*S*AJJ+TWO*S*C*AOD
      A(I,J)=ZER
       A(J,J)=C*C*AJJ+S*S*AII-TWO*S*C*AOD
      A(J,I)=ZER
   11 CONTINUE
      IF(AMAX-EPS)20,20,10
   20 DO 25 I=1,N
   25 E(I)=A(I,I)
      RETURN
      END


