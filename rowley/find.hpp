// $find.hpp 3.0 milbo$
// Derived from code by Henry Rowley http://vasc.ri.cmu.edu/NNFaceDetector.

#ifndef find_hpp
#define find_hpp

void
FindEyesGivenVjFace(DET_PARAMS &DetParams,  // io: on entry has VJ face box
                    const Image &Img,       // in
                    const char sDataDir[]); // in

#endif // find_hpp
