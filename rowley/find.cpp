// $find.cpp 3.0 milbo$
// Derived from code by Henry Rowley http://vasc.ri.cmu.edu/NNFaceDetector.
//
// The original files had the following header:
// By Henry A. Rowley (har@cs.cmu.edu, http://www.cs.cmu.edu/~har)
// Developed at Carnegie Mellon University for the face detection project.
// Code may be used, but please provide appropriate acknowledgements, and let
// har@cs.cmu.edu how you are using it.  Thanks!

#include "stasm.hpp"
#include "list.cpp"

static bool CONF_fSearchAgainForEyes = true; // Re-search at lower res if not found?
                                             // Is true for det params in muct68.shape etc.

// Once we have found an eye, it's best to skip further searching.
// This prevents false detects in the eyebrows which pull the centroid off.
// For an explanatory image, see data/eye-finder-with-false-positives-in-green.bmp

static const int SKIP_WHEN_EYE_FOUND_DIST = 10; // use 0 for no skipping

static const char *NET_0 = "umec";      // neural net names
static const char *NET_1 = "face17c";
static const char *NET_2 = "face18c";
static const char *NET_3 = "eye";

static Image gFaceMask;
static Image gEyeMask;

//-----------------------------------------------------------------------------
// Save detection information in callbackData, which is a list of detections.
// The parameters match the typedef CALLBACK_FUN in search.hpp.

static void
SaveDetections (ClientData callbackData,
     Image *pImg, int x, int y, int width, int height,
     int iLev, double scale, double output)
{
// turn off compiler warning: unused parameter (TODO look into this)
pImg = pImg; width = width; height = height; scale = scale;

List<Detection> *detectionList=(List<Detection> *)callbackData;
detectionList->addLast(new Detection(x, y, iLev, output));
}

//-----------------------------------------------------------------------------
// sFile must be of the form facemask20x20.pgm i.e. name followed by digits
// since we do a basic parse of the filename to extract the mask name

static void
LoadMask (Image &Mask,          // out
          const char sFile[])   // in
{
char sMask[SLEN];

// extract training mask e.g. "facemask20x20.pgm" becomes "facemask"

int i;
for (i = 0; sFile[i] && sFile[i] >= 'A' && sFile[i] <= 'z'; i++)
    sMask[i] = sFile[i];
sMask[i] = 0;

sLoadImage(Mask, sFile);
}

//-----------------------------------------------------------------------------
// Initialize the pNetworkNames used by the Track_FindAllFaces function.
// Also init the globals gFaceMask and gEyeMask.

static void
Init (const char sDataDir[])
{
static bool fInitialized = false;
if (fInitialized)
    return;

// following must be static for InitNetwork
static char s0[SLEN], s1[SLEN], s2[SLEN], s3[SLEN];
sprintf(s0, "%s/%s", sDataDir, NET_0);
sprintf(s1, "%s/%s", sDataDir, NET_1);
sprintf(s2, "%s/%s", sDataDir, NET_2);
sprintf(s3, "%s/%s", sDataDir, NET_3);
const char *pNetworkNames[]={s0, s1, s2, s3};

AllocateNetworks(4);
InitNetwork(0, 1, pNetworkNames);       // network 0: umec
InitNetwork(1, 1, pNetworkNames+1);     // network 1: face17c
InitNetwork(2, 1, pNetworkNames+2);     // network 1: face18c
InitNetwork(3, 1, pNetworkNames+3);     // network 2: eye

char sFace[SLEN];                       // load face mask
logprintf("\n");
sprintf(sFace, "%s/facemask20x20.pgm", sDataDir);
LoadMask(gFaceMask, sFace);

char sMask[SLEN];                       // load eye mask
sprintf(sMask, "%s/eyemask25x15.pgm", sDataDir);
LoadMask(gEyeMask, sMask);

// a consistency check -- prevents overflow in FindEyes()

if (gNetList[3]->nInputs - 1 != gEyeMask.width * gEyeMask.height)
    Err("network \"%s\" number of inputs %d does not match %s size %d",
        pNetworkNames[3], gNetList[3]->nInputs-1,
        sFace, gEyeMask.width * gEyeMask.height);

fInitialized = true;
}

//-----------------------------------------------------------------------------
// Resample an image, according to the following rule: if any of the
// pixels in the source image which correspond to a pixel in the
// destination image is zero, then set the destination pixel to zero.
// Otherwise, set the pixel to 255.
// The idea is that 0 means a pixel location should be scanned, while
// 255 means that a face has already been found and need not be checked
// again.  Destination can be the same as the source.
// Scale < 1 makes the image smaller.
// The scaling factor is determined by the level in the image pyramid,
// and the size of the destination image is controlled by nNewWith and nNewHeight.

static void
ResampleMask (Image &Mask,                                  // out
              const Image &Src,                             // in
              int nNewWidth, int nNewHeight, double Scale)  // in
{
Image TempImg(nNewWidth, nNewHeight);
byte  *buf = TempImg.buf;

FillImage(TempImg, 255);

for (int iy = 0; iy < nNewHeight; iy++)     // scan over lower resolution image
    for (int ix = 0; ix < nNewWidth; ix++)
        {
        int ix1 = int(0.5 + ix / Scale);
        if (ix1 >= Src.width)
            ix1 = Src.width-1;
        int iy1 = int(0.5 + iy / Scale);
        if (iy1 >= Src.height)
            iy1 = Src.height-1;
        int ix2 = int(0.5 + (ix + 1) / Scale);
        if (ix2 >= Src.width)
            ix2 = Src.width-1;
        int iy2 = int(0.5 + (iy + 1) / Scale);
        if (iy2 >= Src.height)
            iy2 = Src.height-1;
        int i, j;
        for (j = iy1; j <= iy2; j++)    // scan over corresponding pixels in hi res
            for (i = ix1; i <= ix2; i++)
                if (!Src(i,j))
                    {                   // if any pixel is zero, make new pixel zero
                    *buf = 0;
                    goto next;
                    }
        next:
            buf++;
        }
Mask = TempImg;
}


//-----------------------------------------------------------------------------
// Figure out the scale at which to run the eye detector.
// The "7" was determined empirically.

static int nGetEyeScale (int nFaceScale)
{
    int nEyeScale = nFaceScale - 7;
    if (nEyeScale >= - 3 && nEyeScale < 0)
        nEyeScale = 0;
    return nEyeScale;
}

//-----------------------------------------------------------------------------
// Search for the eyes in the face and return the two eye positions.
// Left and right are w.r.t.the viewer
// All coordinates are in the scale of Img, which is a scaled down
// version of the original image.

static void FindEyes (
    double *pLex,           // out: left eye position, or INVALID if can't find
    double *pLey,           // out:
    double *pRex,           // out: ditto for right eye
    double *pRey,           // out:
    int xFace, int yFace,   // in: position of top left corner of face box
    double DetWidth,        // in: width of face detector box
    const Image Img,        // in
    const Image &EyeMask,   // in
    int iNet)               // in: neural net index in gNetList
{
*pLex = *pRex = *pRex = *pRey = INVALID;  // assume won't find eyes

int EyeMaskSize  = EyeMask.width * EyeMask.height;
int EyeMaskWidth = EyeMask.width;
int EyeMaskHeight = EyeMask.height;

double xLeft = 0, yLeft = 0, xRight = 0, yRight = 0;
double LeftWeight = 0, RightWeight = 0;

int *pTmpImage = new int[EyeMaskSize];  // window for eye detector

// possible upper-left X positions for the left eye

int startxLeft = iround(xFace);
int endxLeft = iround(xFace + DetWidth/2 - EyeMaskWidth);
if (startxLeft < 0)
    startxLeft = 0;

if (endxLeft > Img.width - EyeMaskWidth)
    endxLeft = Img.width - EyeMaskWidth;

// possible upper-left X positions for the right eye

int startxRight = iround(xFace + DetWidth/2);
int endxRight = iround(xFace + DetWidth - EyeMaskWidth);
if (startxRight < 0)
    startxRight = 0;

if (endxRight > Img.width - EyeMaskWidth)
    endxRight = Img.width - EyeMaskWidth;

// possible upper Y positions for the eyes

int starty = iround(yFace);
int endy = iround(yFace + DetWidth / 4); // exact value 4 is not critical
if (starty < 0)
    starty = 0;
if (endy >= Img.height - EyeMaskHeight)
    endy = Img.height - EyeMaskHeight;

bool fDoneLeft = false;
bool fDoneRight = false;

// start at bottom of face and move up so eyes are before eyebrows,
// thus avoiding false positives on the eyebrows for SKIP_WHEN_EYE_FOUND

for (int y = endy-1; y >= starty; y--)
    {
    if (SKIP_WHEN_EYE_FOUND_DIST != 0)
        {
        // set these if already got the eye, to prevent false detects on eyebrows
        fDoneLeft  = LeftWeight != 0 &&
                        y < yLeft / LeftWeight  - SKIP_WHEN_EYE_FOUND_DIST;

        fDoneRight = RightWeight != 0 &&
                        y < yRight / RightWeight - SKIP_WHEN_EYE_FOUND_DIST;

        if (fDoneLeft && fDoneRight)
            break;
        }
    // Look for right eye on this scan line.  We mirror it so it looks to the
    // net like a right eye because we trained the net on a left eye.

    int i, j, x;

    if (!fDoneRight) for (x = startxRight; x < endxRight; x++)
        {
        int iPixel = 0;
        int Hist[256];
        memset(Hist, 0, sizeof(Hist));

        // copy the window into pTmpImage (using mirror image), and compute
        // the histogram over the entire window

        for (j = 0; j < EyeMaskHeight; j++)
            for (i = EyeMaskWidth - 1; i >= 0; i--) // note: mirror here
                {
                int Pixel = Img(i+x, j+y);
                pTmpImage[iPixel++] = Pixel;
                Hist[Pixel]++;
                }
        // compute cumulative histogram

        int CumHist[256];
        int Sum = 0;
        for (i = 0; i < 256; i++)
            {
            CumHist[i] = Sum;
            Sum += Hist[i];
            }
        int Total = Sum;
        for (i = 255; i >= 0; i--)
            {
            CumHist[i] += Sum;
            Sum -= Hist[i];
            }
        // apply the histogram equalization, and write window to network inputs

        const double Scale = 1.0 / Total;
        ForwardUnit *pUnit = &(gNetList[iNet]->pUnitList[1]);
        for (i = 0; i < EyeMaskSize; i++)
            (pUnit++)->activation = (_FLOAT)(CumHist[pTmpImage[i]] * Scale - 1.0);

        // if the network responds positively, add the detection to centroid

        const double Output = ForwardPass(gNetList[iNet]);
        if (Output > 0)
            {
            RightWeight += Output;
            xRight += Output * (x + EyeMaskWidth / 2);
            yRight += Output * (y + EyeMaskHeight / 2);
            }
        }
    // look for left eye on this scan line

    if (!fDoneLeft) for (x = startxLeft; x < endxLeft; x++)
        {
        int iPixel = 0;
        int Hist[256];
        memset(Hist, 0, sizeof(Hist));

        // copy the window into pTmpImage,  and compute
        // the histogram over the entire window

        for (j = 0; j < EyeMaskHeight; j++)
            for (i = 0; i < EyeMaskWidth; i++)
                {
                int Pixel = Img(i+x, j+y);
                pTmpImage[iPixel++] = Pixel;
                Hist[Pixel]++;
                }

        // compute cumulative histogram

        int CumHist[256];
        int Sum = 0;
        for (i = 0; i < 256; i++)
            {
            CumHist[i] = Sum;
            Sum += Hist[i];
            }
        int Total = Sum;
        for (i = 255; i >= 0; i--)
            {
            CumHist[i] += Sum;
            Sum -= Hist[i];
            }
        // apply the histogram equalization, and write window to network inputs

        const double Scale = 1.0 / Total;
        ForwardUnit *pUnit = &(gNetList[iNet]->pUnitList[1]);
        for (i = 0; i < EyeMaskSize; i++)
            (pUnit++)->activation = (_FLOAT)(CumHist[pTmpImage[i]] * Scale - 1.0);

        // if the network responds positively, add the detection to centroid

        const double Output = ForwardPass(gNetList[iNet]);
        if (Output > 0)
            {
            LeftWeight += Output;
            xLeft += Output * (x + EyeMaskWidth  / 2);
            yLeft += Output * (y + EyeMaskHeight / 2);
            }
        }
    }
// if the left eye was detected at least once, return centroid

if (LeftWeight > 0)
    {
    *pLex = xLeft / LeftWeight;
    *pLey = yLeft / LeftWeight;
    }
// if the right eye was detected at least once, return centroid

if (RightWeight > 0)
    {
    *pRex = xRight / RightWeight;
    *pRey = yRight / RightWeight;
    }
delete[] pTmpImage;
}


//-----------------------------------------------------------------------------
// Set eye coordinates in DetParams to INVALID

static void ZapEyes (DET_PARAMS &DetParams) // io
{
DetParams.lex = DetParams.rex = DetParams.rex = DetParams.rey = INVALID;
}

//-----------------------------------------------------------------------------
// auxilary function for FindEyesGivenVjFace

static void FindEyesGivenVjFace1 (DET_PARAMS *pDet,     // io: eye fields updated
                                  int nEyeScale,
                                  double DetWidth,
                                  const Image &Img)
{
double EyeScale12 = POW12(nEyeScale);
Image ReducedImg(Img);
ReduceImage(ReducedImg, EyeScale12, IM_BILINEAR);
double ScaledDetWidth = DetWidth / EyeScale12;

// cartesian coords
double x = pDet->x / EyeScale12;
double y = pDet->y / EyeScale12;
// opencv coords
x += ReducedImg.width / 2;
y = ReducedImg.height / 2 - y;
// move x.y from center of face box to top right corner of face box
x -= ScaledDetWidth/2;
y -= ScaledDetWidth/2;

// Adjust viola jones y up the image, to correspond to rowley y.
// The conversion factor was discovered by measuring on the training data.
// See regress-vj-to-rowley-width.R. The exact value is not critical.

const double CONF_VjWidthScale = 0.13;
y += CONF_VjWidthScale * ScaledDetWidth;

ZapEyes(*pDet);                     // assume won't find eyes
FindEyes(&pDet->lex, &pDet->ley, &pDet->rex, &pDet->rey,
         iround(x), iround(y), ScaledDetWidth,
         ReducedImg, gEyeMask, 3);  // 3 is iNet i.e. eye.net

if (pDet->lex != INVALID)
    {
    pDet->lex = iround(pDet->lex * EyeScale12 - Img.width/2);
    pDet->ley = iround(Img.height/2 - pDet->ley * EyeScale12);
    }
if (pDet->rex != INVALID)
    {
    pDet->rex = iround(pDet->rex * EyeScale12 - Img.width/2);
    pDet->rey = iround(Img.height/2 - pDet->rey * EyeScale12);
    }
}

//-----------------------------------------------------------------------------
// Get the eye positions given the Rowley Face box parameters
// using the Rowley eye detector.
// Results go into the eye fields in DetParams.

void
FindEyesGivenVjFace (DET_PARAMS &DetParams, // io: on entry has VJ face box
                     const Image &Img,      // in
                     const char sDataDir[]) // in
{
Init(sDataDir);                             // init gFaceMask etc.
ZapEyes(DetParams);                         // assume won't find eyes

// Estimate the Rowley facebox width from the Viola Jones facebox.  The
// params of the formula were determined by linear regression of Rowley
// detector widths on VJ widths.  See regress-vj-to-rowley-width.R.

const double CONF_VjAlpha = -19;
const double CONF_VjBeta = 0.9;

const double DetWidth = CONF_VjAlpha + CONF_VjBeta * DetParams.width;

// Find the amount to scale the image to create the image used for
// searching for eyes.  Since the Rowley facebox width is always 20
// in the scaled image, we can calculate the scaling factor.

int nFaceScale = iround(log(double(DetWidth)/20) / log(1.2));

int nEyeScale = nGetEyeScale(nFaceScale);
if (nEyeScale >= 0)                     // face is big enough?
    {
    FindEyesGivenVjFace1(&DetParams, nEyeScale, DetWidth, Img);
    int nEyes = (DetParams.lex != INVALID) + (DetParams.rex != INVALID);
    if (nEyes != 2 && CONF_fSearchAgainForEyes)
        {
        // eye(s) missing, so search again at the next lower scale

        const int nEyeScale1 = nEyeScale - 1;
        if (nEyeScale1 >= 0)             // face is still big enough?
            {
            DET_PARAMS DetParams1 = DetParams;
            FindEyesGivenVjFace1(&DetParams1, nEyeScale1, DetWidth, Img);
            if (((DetParams1.lex != INVALID) + (DetParams1.rex != INVALID)) > nEyes)
                {
                DetParams = DetParams1; // found more eyes than before, so use them
                nEyeScale = nEyeScale1;
                }
            }
        }
    }
}
