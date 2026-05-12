function[]=Quantitative_measures_w_wo_Denoise(fpathin)

%fpathin: input main folder path which contrain subfolders of all 52 subject
% Each subfolder should contain reference EPI data and CAIPI accelerated
% data (without and with denoising) and the ROI mask

cd(fpathin)
listing =dir(fpathin);
fnames={listing.name};
fnames=fnames(1,3:end);
for i=1:length(fnames)
    i
    fdata =char(fnames(i));
    cd(fdata)
%   load EPI data
%   refrence data: EPI_b, b4 & af indicate before & after denoising
    IEPI=single(niftiread('EPI_b.nii.gz'));

    IEPI12=single(niftiread('EPI12_b4b.nii.gz'));
    IEPI13=single(niftiread('EPI13_b4b.nii.gz'));
    IEPI22=single(niftiread('EPI22_b4b.nii.gz'));  
    
    IEPI12af=single(niftiread('EPI12_afb.nii.gz'));
    IEPI13af=single(niftiread('EPI13_afb.nii.gz'));
    IEPI22af=single(niftiread('EPI22_afb.nii.gz'));  
    
    % load White matter mask
    Mwm=niftiread('WM.nii.gz');
    Mwm12=niftiread('WM12.nii.gz');
    Mwm13=niftiread('WM13.nii.gz');
    Mwm22=niftiread('WM22.nii.gz');
    

    % load lesion probability map
    Mmimosa=niftiread('Leprob.nii.gz');
    Mmimosa12=niftiread('Leprob12.nii.gz');
    Mmimosa13=niftiread('Leprob13.nii.gz');
    Mmimosa22=niftiread('Leprob22.nii.gz');
    

    
    % load vein mask
    Mv=niftiread('vein_mask.nii.gz');
    Mv12=niftiread('vein_mask12.nii.gz');
    Mv13=niftiread('vein_mask13.nii.gz');
    Mv22=niftiread('vein_mask22.nii.gz');
    
    Mask=ones(size(IEPI));
    Mask(IEPIm==0)=0;
    

    IEPI=IEPI./max(IEPI(:));
    IEPI12=IEPI12./max(IEPI12(:));
    IEPI13=IEPI13./max(IEPI13(:));
    IEPI22=IEPI22./max(IEPI22(:));
    IEPI12af=IEPI12af./max(IEPI12af(:));
    IEPI13af=IEPI13af./max(IEPI13af(:));
    IEPI22af=IEPI22af./max(IEPI22af(:));

    % Calculate tissue contrast
    [NCNRl2v,NCNRl2w,NCNRv2w]=Normalized_CNR_CAIPI(IEPI,Mwm,Mmimosa,Mv);
    
    [NCNRl2v12,NCNRl2w12,NCNRv2w12]=Normalized_CNR_CAIPI(IEPI12,Mwm12,Mmimosa12,Mv12);
    [NCNRl2v13,NCNRl2w13,NCNRv2w13]=Normalized_CNR_CAIPI(IEPI13,Mwm13,Mmimosa13,Mv13);
    [NCNRl2v22,NCNRl2w22,NCNRv2w22]=Normalized_CNR_CAIPI(IEPI22,Mwm22,Mmimosa22,Mv22);
    
    [NCNRl2v12af,NCNRl2w12af,NCNRv2w12af]=Normalized_CNR_CAIPI(IEPI12af,Mwm12,Mmimosa12,Mv12);
    [NCNRl2v13af,NCNRl2w13af,NCNRv2w13af]=Normalized_CNR_CAIPI(IEPI13af,Mwm13,Mmimosa13,Mv13);
    [NCNRl2v22af,NCNRl2w22af,NCNRv2w22af]=Normalized_CNR_CAIPI(IEPI22af,Mwm22,Mmimosa22,Mv22);
    
    PSNRref=PSNRmeasurement(IEPI,Mwm,Mv);
    PSNR12=PSNRmeasurement(IEPI12,Mwm12,Mv12);
    PSNR13=PSNRmeasurement(IEPI13,Mwm13,Mv13);
    PSNR22=PSNRmeasurement(IEPI22,Mwm22,Mv22);

    PSNR12af=PSNRmeasurement(IEPI12af,Mwm12,Mv12);
    PSNR13af=PSNRmeasurement(IEPI13af,Mwm13,Mv13);
    PSNR22af=PSNRmeasurement(IEPI22af,Mwm22,Mv22);
    
    [SSIM12,SSIM13,SSIM22,SSIM12af,SSIM13af,SSIM22af]=Quant_ssim(IEPI12,IEPI13,IEPI22,IEPI12af,IEPI13af,IEPI22af);
    
    
    newStr = split(fpathin,'/');
    id=strcat(newStr{end},'-',fnames{i})
    
    Quant.id=id;
    
    Quant.NCNRl2v=NCNRl2v;
    
    Quant.NCNRl2v12=NCNRl2v12;
    Quant.NCNRl2v13=NCNRl2v13;
    Quant.NCNRl2v22=NCNRl2v22;

    Quant.NCNRl2v12af=NCNRl2v12af;
    Quant.NCNRl2v13af=NCNRl2v13af;
    Quant.NCNRl2v22af=NCNRl2v22af;

    Quant.NCNRl2w=NCNRl2w;
    
    Quant.NCNRl2w12=NCNRl2w12;
    Quant.NCNRl2w13=NCNRl2w13;
    Quant.NCNRl2w22=NCNRl2w22;

    Quant.NCNRl2w12af=NCNRl2w12af;
    Quant.NCNRl2w13af=NCNRl2w13af;
    Quant.NCNRl2w22af=NCNRl2w22af;

    Quant.NCNRv2w=NCNRv2w;
    
    Quant.NCNRv2w12=NCNRv2w12;
    Quant.NCNRv2w13=NCNRv2w13;
    Quant.NCNRv2w22=NCNRv2w22;

    Quant.NCNRv2w12af=NCNRv2w12af;
    Quant.NCNRv2w13af=NCNRv2w13af;
    Quant.NCNRv2w22af=NCNRv2w22af;

    Quant.PSNRref=PSNRref;
    Quant.PSNR12=PSNR12;
    Quant.PSNR13=PSNR13;
    Quant.PSNR22=PSNR22;
    Quant.PSNR12af=PSNR12af;
    Quant.PSNR13af=PSNR13af;
    Quant.PSNR22af=PSNR22af;

    Quant.SSIM12=SSIM12;
    Quant.SSIM13=SSIM13;
    Quant.SSIM22=SSIM22;
    Quant.SSIM12af=SSIM12af;
    Quant.SSIM13af=SSIM13af;
    Quant.SSIM22af=SSIM22af;
    
    T = struct2table(Quant);
    % specify the path of excel file
    writetable(T,'/specify_path/Quantmeasures_w_wo_Denoising.xlsx','WriteMode','Append');
    
    
    cd(fpathin)
end




function[NCNRl2v,NCNRl2w,NCNRv2w,Swm,Sv,Sle]=Normalized_CNR_CAIPI(Istk,Wmask,Lmask,Mv)


Istk = double(Istk);
Mask=ones(size(Istk));
Mask(Istk==0)=0;
Wmask=Wmask.*Mask;
Lmask=Lmask.*Mask;
Mv = Mv.*Mask;
[NCNRl2v,NCNRl2w,NCNRv2w,Swm,Sv,Sle]=CNR_LVW(Istk,Wmask,Lmask,Mv);




function[NCNRL2v,NCNRl2w,NCNRv2w,Swm,Sv,Sle]=CNR_LVW(Istk,Mwm,Mls,Mv)

Mwm(Mwm<0.8)=0;
Mwm1=zeros(size(Mwm));
Mwm1(Mwm>=0.8)=1;
Mwm=Mwm1;
Mls(Mls<0.6)=0;
Mls(Mls>=0.6)=1;

indl=find(Mls==1);

indv=find(Mv>0.75);

Maskv=zeros(size(Mv));
Maskv(indv)=1;
Mv=Maskv;
indv=find(Mv==1);
indwm=find(Mwm==1);

ind=setdiff(indwm,indv); 
ind =setdiff(ind,indl);
indl=setdiff(indl,indv);

Swm = mean(Istk(ind));

Sv = mean(Istk(indv));

Sle = mean(Istk(indl));

NCNRL2v = abs(mean(Istk(indl))-mean(Istk(indv)))/abs(mean(Istk(indl))+mean(Istk(indv)));

NCNRl2w = abs(mean(Istk(indl))-mean(Istk(ind)))/abs(mean(Istk(indl))+mean(Istk(ind)));

NCNRv2w = abs(mean(Istk(ind))-mean(Istk(indv)))/(mean(Istk(ind))+mean(Istk(indv)));

function[PSNR]=PSNRmeasurement(Istk,Mwm,Mv)

% Istk: 3D volume of Mag image with slice in 3rd direction
% Mwm & Mv : 3D binary volume of WM & vein (Same dimension as that of Istk)

Istk  = Istk/max(Istk(:));
Mwm(Mwm<0.8)=0;
Mwm1=zeros(size(Mwm));
Mwm1(Mwm1>=0.8)=1;
Mwm=Mwm1;

Mask=Xcrcircularradi(0.5,size(Mwm));
Mwm=Mwm.*Mask;
indv=find(Mv==1);
indwm=find(Mwm==1);
ind=setdiff(indwm,indv);
M=zeros(size(Istk));
M(ind)=1;
M=logical(M);
% Mwm = setdiff(Mwm,Mv);
mean(Istk(M))
std(Istk(M))
SNR = max(Istk(M==1))/std(Istk(M==1));
PSNR = 20*log10(SNR);

function[SSIM12,SSIM13,SSIM22,SSIM12af,SSIM13af,SSIM22af]=Quant_ssim(IEPI12,IEPI13,IEPI22,IEPI12af,IEPI13af,IEPI22af)

%   load EPI data registered to CAIPI data

    IEPIRef12=single(niftiread('EPI_ref12.nii.gz'));
    IEPIRef13=single(niftiread('EPI_ref13.nii.gz'));
    IEPIRef22=single(niftiread('EPI_ref22.nii.gz'));
    
    IEPIRef12=IEPIRef12./max(IEPIRef12(:));
    IEPIRef13=IEPIRef13./max(IEPIRef13(:));
    IEPIRef22=IEPIRef22./max(IEPIRef22(:));

    % Calculate SSIM
    SSIM12=ssim(IEPI12,IEPIRef12);
    SSIM13=ssim(IEPI13,IEPIRef13);
    SSIM22=ssim(IEPI22,IEPIRef22);

    SSIM12af=ssim(IEPI12af,IEPIRef12);
    SSIM13af=ssim(IEPI13af,IEPIRef13);
    SSIM22af=ssim(IEPI22af,IEPIRef22);

    