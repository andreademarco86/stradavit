J/A+A/699/A338         LoTSS DR2 visual classifications          (Horton+, 2025)
================================================================================
Complex morphology and precession indicators of active galactic nuclei jets
in LoTSS DR2.
    Horton M.A., Hardcastle M.J., Miley G.K., Tasse C., Shimwell T.
    <Astron. Astrophys. 699, A338 (2025)>
    =2025A&A...699A.338H        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Galaxy catalogs ; Radio continuum ; Radio sources
Keywords: black hole physics - gravitation - galaxies: active -
          galaxies: general - galaxies: jets - radio continuum: galaxies

Abstract:
    The LOw Frequency ARray Two-metre Sky Survey second data release
    (LoTSS DR2) covers 27% of the northern sky and contains around four
    million radio sources. The development of this catalogue involved a
    large citizen science project (Radio Galaxy Zoo: LOFAR) with more than
    116,000 resolved sources going through visual inspection. We took a
    subset of sources with flux density above 75mJy and an angular size of
    90" or greater, giving a total of 9985 sources or ~10% of the
    visually inspected sources. We classified these by visual inspection
    in terms of broad source type (e.g., Fanaroff-Riley class I or II,
    narrow or wide-angle tail, relaxed double), noticeable features
    (wings, visible jets, banding, filaments etc), environmental features
    (cluster environment, merger, diffuse emission). Our specific aim was
    to search for features linked to jet precession, such as a misaligned
    jet axis, curvature and multiple hotspots. This combination of
    features and morphology allowed us to detect increasingly fine-grained
    sub- populations of interesting or unusual sources. We found that 28%
    of sources showed evidence of one or more precession indicators, which
    could make them candidates for hosting close binary supermassive black
    holes. Potential precession signatures occur in sources of all sizes
    and luminosities in our sample but appear to favour more massive host
    galaxies. Our work greatly expands the sample size and parameter space
    of searches for precession signatures in powerful jetted sources. This
    work also showcases the diversity of large bright radio sources in the
    LOFAR surveys, whether or not precession indicators are present.

Description:
   Visual inspection of large, bright sources from LoTSS DR2.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
catalog.dat      177     9985   Visual classifications and precession indicators
                                 for extended radio sources from LoTSS DR2
--------------------------------------------------------------------------------

See also:
  J/A+A/622/A1   : LOFAR Two-metre Sky Survey DR1 source cat. (Shimwell+, 2019)
  J/A+A/659/A1   : LOFAR Two-metre Sky Survey (LoTSS) DR2 (Shimwell+, 2022)
  J/A+A/678/A151 : LoTSS DR2 optical IDs (Hardcastle+, 2023)

  http://lofar-surveys.org/dr2_release.html : LOFAR surveys website

Byte-by-byte Description of file: catalog.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label      Explanations
--------------------------------------------------------------------------------
   1- 22  A22   ---     Name       Name of the source from the LoTSS DR2 optical
                                    ID catalogue, Cat. J/A+A/678/A151,
                                    ILTJHHMMSS.ss+DDMMSS.s (Source_Name)
  25- 47 E23.17 ---     zbest      ? Best estimate of the source redshift
                                    (z_best)
  49- 71 F23.17 kpc     Size       ? Source physical size in kpc (Size)
  73- 95 E23.17 W/Hz    L144       ? Source radio luminosity at 144 MHz (L_144)
  97-116 F20.16 [Msun]  logMassmed ? Base 10 logarithm of source host galaxy
                                    stellar mass estimate (Mass_median)
     118  I1    ---   f_logMassmed [0/1] Mass quality flag
                                    (do not use if = 0) (flag_mass)
     120  I1    ---     FRI        [0/1] Boolean, source is an FRI (fri)
     122  I1    ---     FRII       [0/1] Boolean, source is an FRII (frii)
     124  I1    ---     Hybrid     [0/1] Boolean, source is a hybrid as
                                    described in the paper (hybrid)
     126  I1    ---     Spiral     [0/1] Boolean, source is a spiral galaxy
                                    (spiral)
     128  I1    ---     Relaxed    [0/1] Boolean, source is a relaxed double
                                    (relaxed)
     130  I1    ---     Cshaped    [0/1] Boolean, source is c-shaped (cshaped)
     132  I1    ---     Sshaped    [0/1] Boolean, source is s-shaped (sshaped)
     134  I1    ---     Misalign   [0/1] Boolean, source is misaligned
                                    (misaligned)
     136  I1    ---     Wings      [0/1] Boolean, source has wings (wings)
     138  I1    ---     Xshaped    [0/1] Boolean, source is x-shaped (xshaped)
     140  I1    ---     Straight   [0/1] Boolean, source has a straight jet
                                    (straight)
     142  I1    ---     MultHSpot  [0/1] Boolean, source has multiple hotspots
                                    in one or both lobes (multihotspots)
     144  I1    ---     Continuous [0/1] Boolean, source has a continuous jet
                                    (continuous)
     146  I1    ---     Banding    [0/1] Boolean, source shows banding structure
                                    (banding)
     148  I1    ---     Onesided   [0/1] Boolean, source is one-sided (onesided)
     150  I1    ---     Restarted  [0/1] Boolean, source shows signs of having
                                    restarted (restarted)
     152  I1    ---     Cluster    [0/1] Boolean, source is a cluster member
                                    (cluster)
     154  I1    ---     Merger     [0/1] Boolean, source is a merger (merger)
     156  I1    ---     Diffuse    [0/1] Boolean, source has diffuse structure
                                    only (diffuse)
     158  I1    ---     Unknown    [0/1] Boolean, source classification is
                                    unknown (unknown)
 160-161  I2    ---     FeatureCT  Total number of features identified
                                    (featurecount)
     163  I1    ---     Hasany1    [0/1] Source has any one precession indicator
                                    (hasanyone)
     165  I1    ---     hasany2    [0/1] Source has any two precession
                                    indicators (hasanytwo)
     167  I1    ---     Hasall     [0/1] Source has all three precession
                                    indicators used in the paper (hasall)
     169  I1    ---     Has1       [0/1] Source has exactly one precession
                                    indicator (has_exactly_1)
     171  I1    ---     Has2       [0/1] Source has exactly two precession
                                    indicators (has_exactly_2)
     173  I1    ---     SE         [0/1] Source shows s symmetry and
                                    misalignment (se)
     175  I1    ---     MS         [0/1] Source shows s symmetry and multiple
                                    hotspots (ms)
     177  I1    ---     EM         [0/1] Source shows multiple hotspots and
                                    misalignment (em)
--------------------------------------------------------------------------------

Acknowledgements:
     Martin Hardcastle, m.j.hardcastle(at)herts.ac.uk

================================================================================
(End)                                        Patricia Vannier [CDS]  28-Apr-2025