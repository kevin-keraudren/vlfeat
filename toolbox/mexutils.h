/** @file   mexutils.h
 ** @author Andrea Vedaldi
 ** @brief  MEX driver support - Declaration
 **
 ** This module provides a set of helper functionalities for writing MEX files.
 **/

/* AUTORIGHTS */

#include"mex.h"
#include<ctype.h>

/** ---------------------------------------------------------------- */
/** @brief Is the array real?
 **
 ** @param A array to test.
 **
 ** An array satisfies the test if:
 ** - The storage class is DOUBLE.
 ** - There is no imaginary part.
 **
 ** @return test result.
 **/
static int
uIsReal(const mxArray* A)
{
  return 
    mxIsDouble(A) && 
    !mxIsComplex(A) ;
}

/** ---------------------------------------------------------------- */
/** @brief Is the array real and scalar?
 **
 ** @param A array to test.
 **
 ** An array is <em>real and scalar</em> if:
 ** - It is real (see ::uIsReal()).
 ** - It as only one element.
 **
 ** @return test result.
 **/
static int
uIsRealScalar(const mxArray* A)
{
  return 
    uIsReal (A) && mxGetNumberOfElements(A) == 1 ;
}

/** ---------------------------------------------------------------- */
/** @brief Is the array a real matrix?
 **
 ** @param A array to test.
 ** @param M number of rows.
 ** @param N number of columns.
 **
 ** The array @a A satisfies the test if:
 ** - It is real (see ::uIsReal()).
 ** - It as two dimensions.
 ** - @a M < 0 or the number of rows is equal to @a M.
 ** - @a N < 0 or the number of columns is equal to @a N.
 **
 ** @return test result.
 **/
static int
uIsRealMatrix(const mxArray* A, int M, int N)
{
  return  
    mxIsDouble(A) &&
    !mxIsComplex(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    (M < 0 || mxGetM(A) == M) &&
    (N < 0 || mxGetN(A) == N) ;   
}

/** ---------------------------------------------------------------- */
/** @brief Is the array real with specified dimensions?
 **
 ** @param A array to check.
 ** @param D number of dimensions.
 ** @param dims dimensions.
 **
 ** The array @a A satisfies the test if:
 ** - It is real (see ::uIsReal()).
 ** - @a ndims < 0 or it has @a ndims dimensions and
 **   - for each element of @a dims, either that element is negative
 **     or it is equal to the corresponding dimesion of the array.
 **
 ** @return test result.
 **/
static int
uIsRealArray(const mxArray* A, int D, const int* dims)
{
  if(!mxIsDouble(A) || mxIsComplex(A))
    return false ;

  if(D >= 0) {
    int d ;
    const int* actual_dims = mxGetDimensions(A) ;

    if(mxGetNumberOfDimensions(A) != D)
      return false ;

    return true  ;
    
    if(dims != NULL) {
      for(d = 0 ; d < D ; ++d) {
        if(dims[d] >= 0 && dims[d] != actual_dims[d])
          return false ;
      }
    }
  }
  return true ;
}

/** ---------------------------------------------------------------- */
/** @brief Is the array a string?
 **
 ** @param A array to test.
 ** @param L string length.
 **
 ** The array @a A satisfies the test if:
 ** - its storage class is CHAR;
 ** - it has two dimensions;
 ** - it has one row;
 ** - @a L < 0 or it has @a L columns.
 **
 ** @return test result.
 **/
static int
uIsString(const mxArray* A, int L)
{
  int M = mxGetM(A) ;
  int N = mxGetN(A) ;

  return 
    mxIsChar(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    M == 1 &&
    (L < 0 || N == L) ;
}


/** ---------------------------------------------------------------- */
/** @brief MEX option */
struct _uMexOption
{
  const char *name ; /**< option name */
  int has_arg ;      /**< has argument? */
  int val ;          /**< value to return */
} ;

/** @brief MEX opion type
 ** @see ::_uMexOption
 **/
typedef struct _uMexOption uMexOption ;

/** ---------------------------------------------------------------- */
/** @brief Case insensitive string comparison
 **
 ** @param s1 fisrt string.
 ** @param s2 second string.
 **
 ** @return 0 if the strings are equal, >0 if the first string is
 ** greater (in lexicographical order) and <0 otherwise.
 **/
int
ustricmp(const char *s1, const char *s2)
{
  while (tolower((unsigned char)*s1) == 
         tolower((unsigned char)*s2))
  {
    if (*s1 == 0)
      return 0;
    s1++;
    s2++;
  }
  return 
    (int)tolower((unsigned char)*s1) - 
    (int)tolower((unsigned char)*s2) ;
}

/** ---------------------------------------------------------------- */
/** @brief Process next option
 **
 ** @param args
 ** @param nargs
 ** @param options
 ** @param next
 **
 ** @return
 **/
static int uNextOption(mxArray const *args[], int nargs, 
                       uMexOption const *options, 
                       int *next, 
                       mxArray const **optarg)
{
  char err_msg [1024] ;
  char name    [1024] ;
  int opt = -1, i, len ;

  if (*next >= nargs) {
    return opt ;
  }
  
  /* check the array is a string */
  if (! uIsString (args [*next], -1)) {
    snprintf(err_msg, sizeof(err_msg),
             "The option name is not a string (argument number %d).",
             *next + 1) ;
    mexErrMsgTxt(err_msg) ;
  }

  /* retrieve option name */
  len = mxGetNumberOfElements (args [*next]) ;
  
  if (mxGetString (args [*next], name, sizeof(name))) {
    snprintf(err_msg, sizeof(err_msg),
             "The option name is too long (argument number %d).", 
             *next + 1) ;
    mexErrMsgTxt(err_msg) ;    
  }
  
  /* advance argumnt list */
  ++ (*next) ;
        
  /* now lookup the string in the option table */
  for (i = 0 ; options[i].name != 0 ; ++i) {    
    if (ustricmp(name, options[i].name) == 0) {
      opt = options[i].val ;
      break ;
    }
  }
  
  /* unknown argument */
  if (opt < 0) {
    snprintf(err_msg, sizeof(err_msg),
             "Unkown option '%s'.", name) ;
    mexErrMsgTxt(err_msg) ;
  }

  /* no argument */
  if (! options [i].has_arg) {
    if (optarg) *optarg = 0 ;
    return opt ;
  }
  
  /* argument */
  if (optarg) *optarg = args [*next] ;
  ++ (*next) ;
  return opt ;  
}