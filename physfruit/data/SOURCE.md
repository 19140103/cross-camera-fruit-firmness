# 吸收基谱来源

## prospect_d_spectra.txt
- 来源: https://github.com/jgomezdans/prosail  (prosail/prospect_d_spectra.txt)
- 原始出处: dataSpec_PDB.m (2017-01-16)
- 文献: Féret, Gitelson, Noble & Jacquemoud (2017). PROSPECT-D: Towards
  modeling leaf optical properties through a complete lifecycle.
  Remote Sensing of Environment.
- 范围: 400-2500 nm, 1 nm 步长, 2101 行
- 列:
  1. wavelength (nm)
  2. refractive index of leaf material
  3. specific absorption coeff. chlorophyll (a+b)  [cm^2 / ug]
  4. specific absorption coeff. carotenoids        [cm^2 / ug]
  5. specific absorption coeff. anthocyanins       [cm^2 / ug]
  6. specific absorption coeff. brown pigments     [arbitrary]
  7. specific absorption coeff. water              [cm^-1]
  8. specific absorption coeff. dry matter         [cm^2 / g]
- 棕色素系数由 Frederic Baret (EMMAH, INRA Avignon) 提供并授权使用

## prospect5_spectra.txt
- 同源，PROSPECT-5 版本（无花青素），备用

## 相机波长覆盖检查
- VIS     (Specim FX10)      397.66-1003.81 nm  -> 400nm 以下 2.34nm 需外插
- VIS_COR (Corning microHSI) 408.03- 901.26 nm  -> 完全覆盖
- NIR     (RedEye 1.7)       919.68-1727.04 nm  -> 完全覆盖
