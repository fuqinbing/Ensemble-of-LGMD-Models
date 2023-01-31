%LGMD2_LGMD1_CascadeNetwork_Dual_Channel(img1, img2, t, parameter)

function [parameter] = LGMD2_LGMD1_CascadeNetwork_Dual_Channel(img1, img2, t, parameter)
    cur_t = mod(t, parameter.timestep) + 1;
    pre_t = mod(t - 1, parameter.timestep) + 1;
    prepre_t = mod(t - 2, parameter.timestep) + 1;
    
    cur_t2 = mod(t, parameter.spiLGMD_timestep) + 1;
    
    parameter.points(cur_t).LGMD2_Photoreceptor = ...
        Highpass(img1, img2, parameter.points(pre_t).LGMD2_Photoreceptor, parameter.delay_hp);
    tmp_ffi = sum(sum(abs(parameter.points(cur_t).LGMD2_Photoreceptor)));
    parameter.points(cur_t).FFI = Lowpass(tmp_ffi/parameter.Ncell, ...
        parameter.points(pre_t).FFI, parameter.points(prepre_t).FFI, parameter.delay_ffi);
    
    Wi_off = max(parameter.points(cur_t).FFI / parameter.Tffi, parameter.Base_off);
    Wi_on = max(2 * Wi_off, parameter.Base_on);
    
    Gauss_kernel = fspecial('gaussian', parameter.size, parameter.sigma);
    Gauss_kernel = Gauss_kernel / sum(sum(Gauss_kernel));
     Blurred = imfilter(parameter.points(cur_t).LGMD2_Photoreceptor, Gauss_kernel, 'same');
    
    parameter.points(cur_t).LGMD2_ONs = Halfwave_ON(Blurred, parameter.points(pre_t).LGMD2_ONs, parameter.dc, parameter.Clip_point);
    parameter.points(cur_t).LGMD2_OFFs = Halfwave_OFF(Blurred, parameter.points(pre_t).LGMD2_OFFs, parameter.dc, parameter.Clip_point);
    
    LGMD2_Exc_ON = conv2(parameter.points(cur_t).LGMD2_ONs, parameter.Kernel_E, 'same');
    LGMD2_Exc_OFF = conv2(parameter.points(cur_t).LGMD2_OFFs, parameter.Kernel_E, 'same');
    
    parameter.points(cur_t).LGMD2_ONs_Delay = Lowpass(LGMD2_Exc_ON, ...
        parameter.points(pre_t).LGMD2_ONs_Delay, parameter.points(prepre_t).LGMD2_ONs_Delay, parameter.delay_ON);
    parameter.points(cur_t).LGMD2_OFFs_Delay = Lowpass(LGMD2_Exc_OFF, ...
        parameter.points(pre_t).LGMD2_OFFs_Delay, parameter.points(prepre_t).LGMD2_OFFs_Delay, parameter.delay_OFF);
    
    LGMD2_Inh_ON =  conv2( parameter.points(cur_t).LGMD2_ONs_Delay, parameter.Kernel_ON, 'same');
    LGMD2_Inh_OFF =  conv2( parameter.points(cur_t).LGMD2_OFFs_Delay, parameter.Kernel_OFF, 'same');
    
    S_on = Competing(LGMD2_Exc_ON, LGMD2_Inh_ON, Wi_on);
    S_off = Competing(LGMD2_Exc_OFF, LGMD2_Inh_OFF, Wi_off);
    
    S_on = max(S_on, 0);
    S_off = max(S_off, 0);
    
    parameter.points(cur_t).LGMD2_Summation = ...
        PolarityChannelSummation(S_on, S_off, parameter.ON_exp, parameter.OFF_exp);
    
    % Second LGMD1 network
    
    LGMD1_Photoreceptor = Highpass2(parameter.points(pre_t).LGMD2_Summation, parameter.points(cur_t).LGMD2_Summation);
    
    parameter.points(cur_t).LGMD1_ONs = Halfwave_ON(LGMD1_Photoreceptor, ...
        parameter.points(pre_t).LGMD1_ONs, parameter.dc, parameter.Clip_point);
    parameter.points(cur_t).LGMD1_OFFs = Halfwave_OFF(LGMD1_Photoreceptor, ...
        parameter.points(pre_t).LGMD1_OFFs, parameter.dc, parameter.Clip_point);
    
    LGMD1_Exc_ON = conv2(parameter.points(cur_t).LGMD1_ONs, parameter.Kernel_E, 'same');
    LGMD1_Exc_OFF = conv2(parameter.points(cur_t).LGMD1_OFFs, parameter.Kernel_E, 'same');
    
    parameter.points(cur_t).LGMD1_ONs_Delay = Lowpass(LGMD1_Exc_ON, ...
        parameter.points(pre_t).LGMD1_ONs_Delay, parameter.points(prepre_t).LGMD1_ONs_Delay, parameter.delay_OFF);
    parameter.points(cur_t).LGMD1_OFFs_Delay = Lowpass(LGMD1_Exc_OFF, ...
        parameter.points(pre_t).LGMD1_OFFs_Delay, parameter.points(prepre_t).LGMD1_OFFs_Delay, parameter.delay_OFF);
    
    LGMD1_Inh_ON =  conv2(parameter.points(cur_t).LGMD1_ONs_Delay, parameter.Kernel_OFF, 'same');
    LGMD1_Inh_OFF =  conv2(parameter.points(cur_t).LGMD1_OFFs_Delay, parameter.Kernel_OFF, 'same');
    
    S_on = Competing(LGMD1_Exc_ON, LGMD1_Inh_ON, Wi_off);
    S_off = Competing(LGMD1_Exc_OFF, LGMD1_Inh_OFF, Wi_off);
    
    S_on = max(S_on, 0);
    S_off = max(S_off, 0);
    
    LGMD1_Summation = PolarityChannelSummation(S_on, S_off,parameter.ON_exp, parameter.OFF_exp);
    
    LGMD1_Compressed = NormalizeContrast(parameter. Kernel_Contrast, LGMD1_Summation, parameter. Ccn);
    
    parameter.points(cur_t).LGMD2_Grouping = conv2(LGMD1_Compressed, parameter.Kernel_G, 'same');
    parameter.points(cur_t).LGMD2_Grouping = thresholding(parameter.points(cur_t).LGMD2_Grouping, parameter.Tsfa );
    
    parameter.points(cur_t).LGMD2_SFA = SFA_Profile(parameter.points(pre_t).LGMD2_SFA, ...
        parameter.points(pre_t).LGMD2_Grouping, parameter.points(cur_t).LGMD2_Grouping, parameter.delay_hp, parameter.Tsfa);
    tmp_sum = sum(sum(parameter.points(cur_t).LGMD2_SFA));
    
    parameter.points(cur_t).Vlgmd = tmp_sum;

    parameter.spiLGMD(cur_t2) = Spiking(parameter.points(pre_t).Vlgmd,...
        parameter.points(cur_t).Vlgmd, parameter.Vth);
    
    network_out = LIF_Neuron(parameter.points(pre_t).Vlgmd, parameter.points(cur_t).Vlgmd,...
        parameter.spiLGMD, cur_t2, parameter.Tin, parameter.Vth) + parameter.Vrest;
    parameter.membrane_potential(t) = network_out;
end

function value = Highpass(pre_in, cur_in, pre_out, a)
    value = a * (pre_out + cur_in - pre_in);
end

function value = Highpass2(pre_in, cur_in)
    value = cur_in - pre_in;
end

function value = Lowpass(cur_in, pre_out, prepre_out, a)
    value = a (1)*  cur_in + a (2)*  pre_out + a (3)*  prepre_out;
end

function value = Halfwave_ON(A, B, dc, Clip_point)
    C = A - Clip_point;
    C(C >= 0) = 1;
    C(C < 0) = 0;
    value = A .* C + dc * B;
end

function value = Halfwave_OFF(A, B, dc, Clip_point)
    C = A - Clip_point;
    C(C >= 0) = 0;
    C(C < 0) = 1;
    value = abs(A) .* C + dc * B;
end

function value = Competing(exc, inh, wi)
    C = exc .* inh;
    C(C >= 0) = 1;
    C(C < 0) = 0;
    value = (exc - inh * wi) .*  C;
end

function value = PolarityChannelSummation(on_exc, off_exc, on_exp, off_exp)
    value = on_exc .^ on_exp + off_exc .^ off_exp;
end

function value = NormalizeContrast(kernel, inSignal, Ccn)
    tmp = conv2(abs(inSignal), kernel, 'same');
    value = tanh(inSignal ./ (Ccn + tmp));
end

function value = thresholding(A, Tsfa)
    C = A - Tsfa;
    C(C >= 0) = 1;
    C(C < 0) = 0;
    value = A .* C ;
end

function cur_out = SFA_Profile(pre_out, pre_in, cur_in, delay, Tsfa)
    diff_in = cur_in - pre_in;
    C = diff_in - Tsfa;
    
    %C是一个判断矩阵，若diff_in大于Tsfa，C中对应元素取0，若diff_in小于等于Tsfa，C中对应元素取1
    C(C > 0) = 2;
    C(C <= 0) = 1;
    C(C == 2) = 0;
    
    cur_tmp = delay * cur_in +delay * (pre_out - pre_in) .* C ;
    cur_out = max(cur_tmp, 0);
end

function value = Spiking(Vpre, Vcur, Vth)
    value = heaviside(Vpre + Vcur - Vth);
end

function value = heaviside(input)
    if (input > 0)
        value =1;
    else
        value =0;
    end
end

function value = LIF_Neuron(Vpre, Vcur, Spi, cur, Tin, Vth)
    pre = 1;
    for t = cur : -1 : 1
        if(Spi(t) == 1)
            pre = t;
            break
        end
    end
    value = Vpre * exp(-1 * (cur - pre) / Tin) + Vcur - Spi(cur) *Vth;
end