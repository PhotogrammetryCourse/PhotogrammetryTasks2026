#include "sift.h"
#include "libutils/rasserts.h"

#include <iostream>
#include <numeric>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Ссылки:
// [lowe04] - Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe, 2004
//
// Примеры реализаций (стоит обращаться только если совсем не понятны какие-то места):
// 1) https://github.com/robwhess/opensift/blob/master/src/sift.c
// 2) https://gist.github.com/lxc-xx/7088609
// 3) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.dispatch.cpp
// 4) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.simd.hpp

namespace {

cv::Mat upsample2x(const cv::Mat& src)
{
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(src.cols * 2, src.rows * 2), 0, 0, cv::INTER_LINEAR);
    return dst;
}

cv::Mat downsample2x(const cv::Mat& src)
{
    int dstW = src.cols / 2;
    int dstH = src.rows / 2;
    cv::Mat dst(dstH, dstW, src.type());
    const int ch = src.channels();

    for (int y = 0; y < dstH; y++) {
        const float* srcRow = src.ptr<float>(y * 2);
        float* dstRow = dst.ptr<float>(y);
        for (int x = 0; x < dstW; x++) {
            std::copy(srcRow + x * 2 * ch, srcRow + x * 2 * ch + ch, dstRow + x * ch);
        }
    }
    return dst;
}

[[maybe_unused]] cv::Mat downsample2x_avg(const cv::Mat& src)
{
    int dstW = src.cols / 2;
    int dstH = src.rows / 2;
    cv::Mat dst(dstH, dstW, src.type());

    for (int y = 0; y < dstH; y++) {
        const float* r0 = src.ptr<float>(y * 2);
        const float* r1 = src.ptr<float>(y * 2 + 1);
        float* dstRow = dst.ptr<float>(y);
        for (int x = 0; x < dstW; x++) {
            dstRow[x] = (r0[x * 2] + r0[x * 2 + 1] + r1[x * 2] + r1[x * 2 + 1]) * 0.25f;
        }
    }
    return dst;
}
}

cv::Mat phg::toGray32F(const cv::Mat& img)
{
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = img;
    }

    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F, 1.0 / 255.0);
    return gray_float;
}

std::vector<phg::SIFT::Octave> phg::buildOctaves(const cv::Mat& img, const phg::SIFTParams& p, int verbose_level)
{
    const int s = p.n_octave_layers;
    const double sigma0 = p.sigma;
    // взятое с потолка значение блюра который уже есть в картинке. используем для того, чтобы не так сильно блюрить базовую картинку и не терять лишний раз фичи
    // upd: хотя llm не соглашается со "взятое с потолка":
    //        It is strictly not taken from the ceiling. sigma=0.5 is the theoretical minimum blur needed to prevent aliasing (Nyquist frequency)
    //        when sampling a continuous signal into a discrete grid. It is a mathematically grounded assumption for digital images.
    // общая идея в том, что у нас есть какой-то сигнал реального мира, и есть входное изображение
    //   сигнал реального мира: потенциально высочайшего разрешения, можем зумиться почти до молекул, сигма почти нулевая
    //   сигнал с камеры, входное изображение: было произведено усреднение хотя бы по отдельным пикселям матрицы камеры, что соответствует сигме в полпикселя
    const double sigma_nominal = p.upscale_first ? 1.0 /*2x от неапскейленного*/ : 0.5;
    const int n_layers = s + 3; // нужно +2 слоя для того чтобы крайних было по соседу для поиска максимума в scale space, и еще +1 слой, чтобы получить s DoG слоев (DoG = разность двух)

    int n_octaves = std::max(1, (int)std::round(std::log2(std::min(img.cols, img.rows))) - 3); // не даунскейлим дальше размера картинки в 16 пикселей, там уже не имеет смысла что-то детектировать

    cv::Mat base;
    double sigma_base = std::sqrt(sigma0 * sigma0 - sigma_nominal * sigma_nominal); // можно использовать дальше как идею для инкрементального блюра слоев
    cv::GaussianBlur(img, base, cv::Size(), sigma_base, sigma_base);

    std::vector<phg::SIFT::Octave> octaves(n_octaves);

    double pg = std::pow(2.0, 1.0/s);
    for (int o = 0; o < n_octaves; o++) {
        phg::SIFT::Octave& oct = octaves[o];
        oct.layers.resize(n_layers);

        oct.layers[0] = base.clone();

        // для простоты в каждой октаве будем каждый раз блюрить базовую картинку с полной сигмой
        //  можно подумать, как сделать эффективнее - для построения n+1 слоя доблюревать уже поблюренный n-ый слой, так чтобы в итоге получилась такая же сигма
        //  это будет немного быстрее, тк нужно более маленькое ядро свертки на каждый шаг

        for (int i = 1; i < n_layers; i++) {

            double sigma_layer = sigma0 * std::pow(pg, i);
            double sigma_layer_prev = sigma0 * std::pow(pg, i-1);
            double sigma_result = std::sqrt(sigma_layer*sigma_layer - sigma_layer_prev*sigma_layer_prev);
            cv::GaussianBlur(oct.layers[i-1], oct.layers[i], cv::Size(), sigma_result, sigma_result);
        }

        // подготавливаем базовый слой для следующей октавы
        if (o + 1 < n_octaves) {
            // используется в opencv, формула для пересчета ключевых точек: pt_upscaled = 2^o * pt_downscaled
            base = downsample2x(oct.layers[s]);
            // можно использовать и downsample2x_avg(oct.layers[s]), это позволяет потом заапскейлить слои обратно до оригинального разрешения без сдвига
            // но потребуется везде изменить формулу для пересчета ключевых точек: pt_upscaled = (pt_downscaled + 0.5) * 2^o - 0.5

            if (verbose_level)
                std::cout << "new octave base size: " << base.size().width << std::endl;
        }
    }

    return octaves;
}

std::vector<phg::SIFT::Octave> phg::buildDoG(const std::vector<phg::SIFT::Octave>& octaves, const phg::SIFTParams& p, int verbose_level)
{
    std::vector<phg::SIFT::Octave> dog(octaves.size());

    for (size_t o = 0; o < octaves.size(); o++) {
        const phg::SIFT::Octave& octave = octaves[o];
        dog[o].layers.resize(octave.layers.size() - 1);

        for (size_t d = 0; d < octave.layers.size() - 1; d++) {
            cv::Mat delta;
            cv::subtract(octave.layers[d+1], octave.layers[d], delta);
            dog[o].layers[d] = delta;

        }
    }

    return dog;
}

std::vector<cv::KeyPoint> phg::findScaleSpaceExtrema(const std::vector<phg::SIFT::Octave>& dog, const phg::SIFTParams& params, int verbose_level)
{
    const int s = params.n_octave_layers;
    const double sigma0 = params.sigma;
    const double contrast_threshold = params.contrast_threshold;
    const double edge_threshold = params.edge_threshold;

    // чем больше слоев в октаве, тем меньше разница между ними -> компенсируем порог
    const auto thresh = (float)(contrast_threshold / s);

    const int border = 5;

    const int max_interp_steps = 5;

    const int first_octave = params.upscale_first ? -1 : 0;

    std::vector<cv::KeyPoint> keypoints;

    for (int o = 0; o < (int)dog.size(); o++) {
        int real_octave = o + first_octave;

        const std::vector<cv::Mat>& dog_layers = dog[o].layers;
        const int n_dog_layers = (int)dog_layers.size();
        rassert(n_dog_layers == s + 2, 2138971238612312);

        // итерируемся по внутренним слоям пирамиды, у нас всегда есть предыдущий и следующий сосед
        for (int layer = 1; layer <= s; layer++) {
            const cv::Mat& dog_curr = dog_layers[layer];
            const cv::Mat& dog_prev = dog_layers[layer - 1];
            const cv::Mat& dog_next = dog_layers[layer + 1];

            int rows = dog_curr.rows, cols = dog_curr.cols;

            for (int y = border; y < rows - border; y++) {
                const float* c = dog_curr.ptr<float>(y);
                const float* cp = dog_curr.ptr<float>(y - 1);
                const float* cn = dog_curr.ptr<float>(y + 1);

                const float* p = dog_prev.ptr<float>(y);
                const float* pp = dog_prev.ptr<float>(y - 1);
                const float* pn = dog_prev.ptr<float>(y + 1);

                const float* n = dog_next.ptr<float>(y);
                const float* np = dog_next.ptr<float>(y - 1);
                const float* nn = dog_next.ptr<float>(y + 1);

                for (int x = border; x < cols - border; x++) {
                    float val = c[x];

                    // cascade filtering: предварительная слабая проверка, отбрасываем точку если она не дает хотя бы половину от требуемого порога,
                    //  в надежде что потом после оптимизации порог будет пробит
                    //  так мы и отбрасываем кучу мусора и не слишком строго судим точки которые пока еще не оптимальные
                    if (std::abs(val) < thresh * 0.5f)
                        continue;

                    bool is_max = true, is_min = true;
                    auto check = [&](float v) {
                        if (v >= val)
                            is_max = false;
                        if (v <= val)
                            is_min = false;
                    };

                    check(cp[x-1]);
                    check(cp[x]);
                    check(cp[x+1]);
                    check(c[x-1]);
                    check(c[x+1]);
                    check(cn[x-1]);
                    check(cn[x]);
                    check(cn[x+1]);

                    if (!is_max && !is_min)
                        continue;

                    check(pp[x-1]);
                    check(pp[x]);
                    check(pp[x+1]);
                    check(p[x-1]);
                    check(p[x]);
                    check(p[x+1]);
                    check(pn[x-1]);
                    check(pn[x]);
                    check(pn[x+1]);

                    if (!is_max && !is_min)
                        continue;

                    check(np[x-1]);
                    check(np[x]);
                    check(np[x+1]);
                    check(n[x-1]);
                    check(n[x]);
                    check(n[x+1]);
                    check(nn[x-1]);
                    check(nn[x]);
                    check(nn[x+1]);

                    if (!is_max && !is_min)
                        continue;

                    int xi = x, yi = y, li = layer;

                    for (int step = 0; step < max_interp_steps; step++) {
                        const cv::Mat& cL = dog_layers[li];
                        const cv::Mat& pL = dog_layers[li - 1];
                        const cv::Mat& nL = dog_layers[li + 1];

                        float resp_center = cL.at<float>(yi, xi);

                        // градиент
                        float dx = (cL.at<float>(yi, xi + 1) - cL.at<float>(yi, xi - 1)) * 0.5f;
                        float dy = (cL.at<float>(yi + 1, xi) - cL.at<float>(yi - 1, xi)) * 0.5f;
                        float ds = (nL.at<float>(yi, xi) - pL.at<float>(yi, xi)) * 0.5f;

                        // гессиан
                        float dxx, dxy, dyy, dxs, dys, dss;
                        dxx = cL.at<float>(yi, xi + 1) + cL.at<float>(yi, xi - 1) - 2.f * resp_center;
                        dyy = cL.at<float>(yi + 1, xi) + cL.at<float>(yi - 1, xi) - 2.f * resp_center;
                        dss = nL.at<float>(yi, xi) + pL.at<float>(yi, xi) - 2.f * resp_center;
                        dxy = (cL.at<float>(yi + 1, xi + 1) - cL.at<float>(yi + 1, xi - 1) - cL.at<float>(yi - 1, xi + 1) + cL.at<float>(yi - 1, xi - 1)) * 0.25f;
                        dxs = (nL.at<float>(yi, xi + 1) - nL.at<float>(yi, xi - 1) - pL.at<float>(yi, xi + 1) + pL.at<float>(yi, xi - 1)) * 0.25f;
                        dys = (nL.at<float>(yi + 1, xi) - nL.at<float>(yi - 1, xi) - pL.at<float>(yi + 1, xi) + pL.at<float>(yi - 1, xi)) * 0.25f;

                        cv::Matx33f H(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);

                        cv::Vec3f g(dx, dy, ds);

                        // в нашей точке производная (градиент) еще не равна нулю (т.к. еще мы скорее всего не точно в оптимуме)
                        // хотим найти такой offset, где ноль производной. в предположении что оптимизируемая функция это парабола, ищем корни ее производной, линейной функции
                        // grad(x + offset) = grad(x) + grad'(x) * offset = grad(x) + hessian(x) * offset = 0
                        // hessian(x) * offset = -grad(x) // линейная система. можно решить специализированным решателем либо просто найти обратную матрицу гессиана и домножить на минус градиент
                        // offset = -hessian(x)^-1 * grad(x)

                        cv::Vec3f offset;
                        if (!cv::solve(H, -g, offset, cv::DECOMP_LU))
                            break;

                        if (std::abs(offset[0]) < 0.5f && std::abs(offset[1]) < 0.5f && std::abs(offset[2]) < 0.5f) {

                            // функцию респонза оптимизировали как параболу: D(x + offset) = D(x) + grad(x) * offset + 1/2 * offset_transposed * hessian(x) * offset
                            // подставляем hessian(x) * offset = -grad(x): D(x + offset) = D(x) + grad(x) * offset - 1/2 * offset_transposed * grad = D(x) + 1/2 * grad(x) * offset
                            float response_optimized = resp_center + 0.5f * g.dot(offset);
                            if (std::abs(response_optimized) < thresh)
                                break;

                            // фильтрация по зацепистости
                            if (params.enable_edge_like_filtering) {
                                // из линейной алгебры, сумма диагональных элементов матрицы (след) равна сумме собственных чисел
                                // определитель матрицы равен произведению собственных чисел
                                // в случае гессиана (пространственной части: (dxx dxy, dxy, dyy)), собственные числа lambda1, lambda2 - силы кривизны в направлении максимальной кривизны и в ортогональном
                                float trace = dxx + dyy;
                                float det = dxx * dyy - dxy * dxy;
                                if (det <= 0)
                                    break; // если произведение кривизн отрицательное, то мы находимся в седловой точке, а не в максимуме/минимуме. если нулевое, то это ровная граница вообще

                                // если граница незацепистая = грань, то одна кривизна сильно больше чем другая. хотим, чтобы обе кривизны были примерно сопоставимы
                                // тогда их отношение r = lambda1/lambda2 будет не очень большим
                                // если расписать trace * trace / det через r, то получится (r + 1) ^ 2 / r
                                // функция растущая по r, так что если наше фактическое значение trace * trace / det выше (r + 1) ^ 2 / r, то и наше отношение кривизн больше порога, значит плохая зацепистость
                                // и просто как интуиция, при больших r это выражение просто до r сокращается

                                // в итоге получается что порог edge_threshold в отличие от response_threshold наоборот, чем больше тем расслабленнее
                                float r = edge_threshold;
                                if (trace * trace / det >= (r + 1.0) * (r + 1.0) / r)
                                    break;
                            }

                            // скейлим координаты точек обратно до родных размеров картинки
                            // !!! если выбираем при даунскейле другую политику, с усреднением вместо ресемплинга, то надо здесь применять формулу со сдвигами на полпикселя
                            float scale = (real_octave >= 0) ? (float)(1 << real_octave) : (1.f / (float)(1 << (-real_octave)));
                            float real_x = (xi + offset[0]) * scale;
                            float real_y = (yi + offset[1]) * scale;
                            float real_layer = li + offset[2];

                            if (!params.enable_subpixel_localization) {
                                real_x = x * scale;
                                real_y = y * scale;
                                real_layer = layer;
                            }

                            float kp_sigma = (float)(sigma0 * std::pow(2.0, (double)real_layer / s) * scale);

                            cv::KeyPoint kp;
                            kp.pt.x = real_x;
                            kp.pt.y = real_y;
                            kp.size = kp_sigma * 2.f; // диаметр
                            kp.octave = real_octave;
                            kp.class_id = li; // в настоящей opencv имплементации и слой и октава запихиваются в поле octave битовыми операциями
                            kp.response = std::abs(response_optimized);
                            keypoints.push_back(kp);
                            break;
                        }

                        // это на случай если не зашли в предыдущий if: если оптимизированная точка вылетела за границы нашего пикселя, то делаем еще шаг
                        // идея в том, что если максимум реально там (а у нас же неидеальная парабола), то оптимизировав в той клеточке еще раз, если мы получим маленький сдвиг, то подтвердим минимум и успокоимся
                        // а если снова вылетим из пикселя, то либо поищем минимум еще, либо устанем и забьем
                        xi += cvRound(offset[0]);
                        yi += cvRound(offset[1]);
                        li += cvRound(offset[2]);

                        if (li < 1 || li > s || xi < border || xi >= cols - border || yi < border || yi >= rows - border)
                            break;
                    }
                }
            }
        }

        if (verbose_level)
            std::cout << "octave " << o << ": " << keypoints.size() << " keypoints so far" << std::endl;
    }

    if (verbose_level)
        std::cout << "total keypoints: " << keypoints.size() << std::endl;

    return keypoints;
}

std::vector<cv::KeyPoint> phg::computeOrientations(const std::vector<cv::KeyPoint>& kpts, const std::vector<phg::SIFT::Octave>& octaves, const phg::SIFTParams& params, int verbose_level)
{
    const int s = params.n_octave_layers;
    const double sigma0 = params.sigma;
    const int n_bins = params.orient_nbins;
    const double peak_ratio = params.orient_peak_ratio;

    std::vector<float> histogram(n_bins);

    std::vector<cv::KeyPoint> oriented_kpts;

    const int first_octave = params.upscale_first ? -1 : 0;

    for (const cv::KeyPoint& kp : kpts) {
        int layer = kp.class_id;
        int real_octave = kp.octave;
        int o = real_octave - first_octave; // индекс в массиве octaves

        const cv::Mat& img = octaves[o].layers[layer];

        float scale = (real_octave >= 0) ? (float)(1 << real_octave) : (1.f / (float)(1 << (-real_octave)));
        float x = kp.pt.x / scale;
        float y = kp.pt.y / scale;

        // найдем радиус ключевой точки в координатах ее октавы
        float kp_sigma_octave = (float)(sigma0 * std::pow(2.0, (double)layer / s));
        float sigma_win = 1.5f * kp_sigma_octave; // цитата из lowe: "Each sample added to the histogram is weighted by its gradient magnitude and by a Gaussian-weighted circular window with a σ that is 1.5 times that of the scale of the keypoint."
        int radius = (int)std::round(3.f * sigma_win);

        int xi = (int)std::round(x);
        int yi = (int)std::round(y);

        if (xi - radius <= 0 || xi + radius >= img.cols - 1 || yi - radius <= 0 || yi + radius >= img.rows - 1)
            continue;

        histogram.assign(n_bins, 0.0);

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int px = xi + dx;
                int py = yi + dy;

                // градиент
                float gx = img.at<float>(py, px + 1) - img.at<float>(py, px - 1);
                float gy = img.at<float>(py + 1, px) - img.at<float>(py - 1, px);

                float mag = std::sqrt(gx*gx + gy*gy);
                float angle = std::atan2(gy, gx); // [-pi, pi]

                float angle_deg = angle * 180.f / (float) CV_PI;
                if (angle_deg < 0.f) angle_deg += 360.f;

                // гауссово взвешивание голоса точки с затуханием к краям
                float weight = std::exp(-(dx*dx + dy*dy) / (2.f * sigma_win * sigma_win));
                if (!params.enable_orientation_gaussian_weighting) {
                    weight = 1.f;
                }

                // голосуем в гистограмме направлений. находим два ближайших бина и гладко распределяем голос между ними
                //  в таком случае, голос попавший близко к границе между бинами, проголосует поровну за оба бина
                float bin = angle_deg * n_bins / 360.f;
                if (bin >= n_bins) bin -= n_bins;
                int bin0 = (int) bin;
                int bin1 = (bin0 + 1) % n_bins;

                float frac = bin - bin0;
                if (!params.enable_orientation_bin_interpolation) {
                    frac = 0.0f;
                }

                histogram[bin0] += mag*weight*(1.0f - frac);
                histogram[bin1] += mag*weight*frac;
            }
        }

        // немного сгладим гистограмму: сделаем несколько проходов box-blur (повторный box blur приближает гауссово размытие)
        for (int iter = 0; iter < 6; iter++) {
            float first = histogram[0];
            float prev = histogram[n_bins - 1];
            for (int i = 0; i < n_bins - 1; i++) {
                float tmp = histogram[i];
                histogram[i] = (prev + histogram[i] + histogram[i + 1]) / 3.f;
                prev = tmp;
            }
            histogram[n_bins - 1] = (prev + histogram[n_bins - 1] + first) / 3.f;
        }

        // находим порог: все максимумы сильнее чем peak_ratio * max_val будут приняты и сгенерирована точка
        //  таким образом, на одну задетектированную точку может быть порождено несколько ориентированных точек, если сложно определить однозначно, куда она была направлена
        float max_val = *std::max_element(histogram.begin(), histogram.end());

        for (int i = 0; i < n_bins; i++) {
            int prev = (i + n_bins - 1) % n_bins;
            int next = (i + 1) % n_bins;

            // если локальный максимум и респонз больше порога
            if (histogram[i] > histogram[prev] && histogram[i] > histogram[next] && histogram[i] >= peak_ratio * max_val) {
                float left = histogram[prev];
                float center = histogram[i];
                float right = histogram[next];

                //  хотим найти угол дескриптора точнее = зафитить параболу по трем точкам (i-1, left), (i, center), (i+1, right)
                //  у параболы f(x) = ax^2 + bx + c, экстремум в точке x = offset = -b/(2a)
                //  f(-1) = left, f(0) = center, f(1) = right
                //  f(0) = c = center
                //  f(1) = a + b + c = right
                //  f(-1) = a - b + c = left
                //  f(1) + f(-1) = 2a + 2c -> a = (left + right - 2 * center) / 2
                //  f(1) - f(-1) = 2b -> b = (right - left) / 2

                float a = (left + right - 2*center)*0.5f;
                float b = (right - left) * 0.5f;
                float offset = 0.0f;
                if (a != 0) {
                    offset = std::min(std::max(-b / (2*a), -0.5f), 0.5f);
                }
                if (!params.enable_orientation_subpixel_localization || a == 0) {
                    offset = 0.f;
                }

                float bin_real = i + offset;
                if (bin_real < 0.f) bin_real += n_bins;
                if (bin_real >= n_bins) bin_real -= n_bins;

                float angle = bin_real * 360.f / n_bins;

                cv::KeyPoint new_kp = kp;
                new_kp.angle = angle;
                oriented_kpts.push_back(new_kp);
            }
        }
    }

    if (verbose_level)
        std::cout << "orientations: " << kpts.size() << " -> " << oriented_kpts.size() << " keypoints" << std::endl;

    return oriented_kpts;
}

// дескриптор подсчитывается по более широкой окрестности. если она выходит за границы изображения, точка может быть отброшена, в результате чего массив kpts может измениться
std::pair<cv::Mat, std::vector<cv::KeyPoint>> phg::computeDescriptors(const std::vector<cv::KeyPoint>& kpts, const std::vector<phg::SIFT::Octave>& octaves, const phg::SIFTParams& params, int verbose_level)
{
    const int s = params.n_octave_layers;
    const double sigma0 = params.sigma;

    // будем считать дескриптор внутри патча вокруг ключевой точки
    // структура патча: 4x4 сетка, в каждой клетке гистограмма градиентов на 8 бинов
    const int n_spatial_bins = 4;
    const int n_orient_bins = 8;
    const int n_dims = n_spatial_bins * n_spatial_bins * n_orient_bins; // 128

    // размер одной клетки патча в сигмах. всего размер контекста для одного дексриптора = n_spatial_bins * spatial_bin_width_sigmas сигм
    const float spatial_bin_width_sigmas = 3.f; // в сигмах

    const float mag_cap = 0.2f;

    std::vector<cv::KeyPoint> valid_kpts;
    cv::Mat descriptors;

    const int first_octave = params.upscale_first ? -1 : 0;

    for (const cv::KeyPoint& kp : kpts) {
        int layer = kp.class_id;
        int real_octave = kp.octave;
        int o = real_octave - first_octave; // индекс в массиве octaves

        const cv::Mat& img = octaves[o].layers[layer];

        float scale = (real_octave >= 0) ? (float)(1 << real_octave) : (1.f / (float)(1 << (-real_octave)));
        float x = kp.pt.x / scale;
        float y = kp.pt.y / scale;

        float kp_sigma_octave = (float)(sigma0 * std::pow(2.0, (double)layer / s));

        // размер патча в котором считаем дескриптор в пикселях октавы
        float spatial_bin_width = spatial_bin_width_sigmas * kp_sigma_octave;

        // изначально ширина дескриптора = spatial_bin_width * n_spatial_bins, но берем с запасом:
        // * sqrt(2) для того, чтобы можно было посемплировать патч даже повернутый на 45 градусов ромбиком
        // * +1 в скобках чтобы можно было семплировать градиенты (а еще зачем?)
        float half_width = 0.5f * spatial_bin_width * (n_spatial_bins + 1) * std::sqrt(2.f);
        int radius = (int)std::round(half_width);

        int xi = (int)std::round(x);
        int yi = (int)std::round(y);

        if (xi - radius <= 0 || xi + radius >= img.cols - 1 || yi - radius <= 0 || yi + radius >= img.rows - 1)
            continue;

        float kp_angle_rad = kp.angle * (float)CV_PI / 180.f;
        float cos_a = std::cos(kp_angle_rad);
        float sin_a = std::sin(kp_angle_rad);

        // для гауссового взвешивания: затухающий вклад градиентов с краев картинки
        float sigma_desc = (float)n_spatial_bins * 0.5f;

        std::vector<float> desc(n_dims, 0.f);

        // семплируем градиенты и кладем в гистограммы
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int px = xi + dx;
                int py = yi + dy;

                float rot_x = (cos_a * dx + sin_a * dy) / spatial_bin_width;
                float rot_y = (-sin_a * dx + cos_a * dy) / spatial_bin_width;

                // подсчет пространственного бина
                //       бин 0       бин 1   |     бин 2       бин 3
                // [-----------][-----------] [-----------][-----------]
                //                           ^
                //                           центр ключевой точки (rot_x = rot_y = 0)
                // центр нулевого бина в координатах rot находится в точке (-1.5, -1.5), а после сдвига перемещается в точку (0, 0), что удобно для индексирования
                float bin_x = rot_x + n_spatial_bins * 0.5f - 0.5f;
                float bin_y = rot_y + n_spatial_bins * 0.5f - 0.5f;

                if (bin_x < -1.f || bin_x >= (float)n_spatial_bins || bin_y < -1.f || bin_y >= (float)n_spatial_bins)
                    continue;

                // градиент (потом все равно будем все нормализовывать, так что можно не нормировать здесь)
                float gx = img.at<float>(py, px + 1) - img.at<float>(py, px - 1);
                float gy = img.at<float>(py + 1, px) - img.at<float>(py - 1, px);

                float mag = std::sqrt(gx * gx + gy * gy);
                float angle = std::atan2(gy, gx);

                // инвариантность к повороту: повернем направление градиента на угол ключевой точки
                float angle_invariant = angle - kp_angle_rad;
                while (angle_invariant < 0.f)
                    angle_invariant += (float)CV_2PI;
                while (angle_invariant >= (float)CV_2PI)
                    angle_invariant -= (float)CV_2PI;

                // подсчет бина в гистограммке градиентов внутри пространственного бина
                float bin_o = angle_invariant * n_orient_bins / CV_2PI;
                if (bin_o >= n_orient_bins)
                    bin_o -= n_orient_bins;

                // семплы вблизи края патча взвешиваем с меньшим весом
                float weight = std::exp(-(rot_x * rot_x + rot_y * rot_y) / (2.f * sigma_desc * sigma_desc));
                if (!params.enable_descriptor_gaussian_weighting) {
                    weight = 1.f;
                }
                float weighted_mag = mag * weight;

                if (params.enable_descriptor_bin_interpolation) {
                    // размажем вклад weighted_mag по пространственным бинам и по бинам гистограммок трилинейной интерполяцией

                    int ix0 = (int)std::floor(bin_x);
                    int iy0 = (int)std::floor(bin_y);
                    int io0 = (int)std::floor(bin_o);

                    float fx = bin_x - ix0;
                    float fy = bin_y - iy0;
                    float fo = bin_o - io0;

                    for (int diy = 0; diy <= 1; diy++) {
                        int iy = iy0 + diy;
                        if (iy < 0 || iy >= n_spatial_bins)
                            continue;
                        float wy = (diy == 0) ? (1.f - fy) : fy;

                        for (int dix = 0; dix <= 1; dix++) {
                            int ix = ix0 + dix;
                            if (ix < 0 || ix >= n_spatial_bins)
                                continue;
                            float wx = (dix == 0) ? (1.f - fx) : fx;

                            for (int dio = 0; dio <= 1; dio++) {
                                int io = (io0 + dio) % n_orient_bins;
                                if (io < 0)
                                    io += n_orient_bins;
                                float wo = (dio == 0) ? (1.f - fo) : fo;

                                int idx = (iy * n_spatial_bins + ix) * n_orient_bins + io;
                                desc[idx] += weighted_mag * wx * wy * wo;
                            }
                        }
                    }
                } else {
                    int ix_nearest = (int)std::round(bin_x);
                    int iy_nearest = (int)std::round(bin_y);
                    int io_nearest = (int)std::round(bin_o) % n_orient_bins;

                    if (ix_nearest >= 0 && ix_nearest < n_spatial_bins && iy_nearest >= 0 && iy_nearest < n_spatial_bins) {
                        int idx = (iy_nearest * n_spatial_bins + ix_nearest) * n_orient_bins + io_nearest;
                        desc[idx] += weighted_mag;
                    }
                }
            }
        }

        // нормализуем дескриптор до единичной l2 длины
        float norm = 0.f;
        for (float v : desc)
            norm += v * v;
        norm = std::sqrt(norm) + 1e-7f;
        for (float& v : desc)
            v /= norm;

        // грохнем слишком большие градиенты и ренормализуем
        // таким образом один выброс не потянет за собой весь дескриптор и в будущем расстояние с похожим соседом не вырастет сильно
        for (float& v : desc)
            v = std::min(v, mag_cap);

        norm = 0.f;
        for (float v : desc)
            norm += v * v;
        norm = std::sqrt(norm) + 1e-7f;
        for (float& v : desc)
            v /= norm;

        if (descriptors.empty()) {
            descriptors.create(0, n_dims, CV_32F);
        }

        cv::Mat row(1, n_dims, CV_32F, desc.data());
        descriptors.push_back(row.clone());
        valid_kpts.push_back(kp);
    }

    if (verbose_level)
        std::cout << "descriptors: " << kpts.size() << " -> " << valid_kpts.size() << " keypoints (some discarded due to border)" << std::endl;

    return { descriptors, valid_kpts };
}

std::vector<cv::KeyPoint> phg::selectTopKeypoints(const std::vector<cv::KeyPoint>& kpts, const phg::SIFTParams& params, int verbose_level)
{
    if (params.nfeatures <= 0 || (int)kpts.size() <= params.nfeatures) {
        return kpts;
    }

    int nfeatures = params.nfeatures;

    std::vector<int> idx(kpts.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + nfeatures, idx.end(), [&kpts](int a, int b) { return std::abs(kpts[a].response) > std::abs(kpts[b].response); });
    idx.resize(nfeatures);
    std::sort(idx.begin(), idx.end());

    std::vector<cv::KeyPoint> sel_kpts(nfeatures);
    for (int i = 0; i < nfeatures; ++i) {
        sel_kpts[i] = kpts[idx[i]];
    }

    if (verbose_level)
        std::cout << "retained top " << nfeatures << " keypoints by response" << std::endl;

    return sel_kpts;
}

void phg::SIFT::detectAndCompute(const cv::Mat& img, const cv::Mat& mask, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) const
{
    rassert(mask.empty(), 911738571854310); // not implemented, parameter added to match interface with opencv sift implementation

    saveImg("00_input.jpg", img);

    cv::Mat gray = toGray32F(img);
    saveImg("01_gray.png", gray);

    if (p.upscale_first) {
        auto prev_size = gray.size();
        gray = upsample2x(gray);
        if (verbose_level)
            std::cout << "upscaled image from " << prev_size.width << "x" << prev_size.height << " to " << gray.cols << "x" << gray.rows << std::endl;
        saveImg("01b_gray_upscaled.png", gray);
    }

    std::vector<Octave> octaves = buildOctaves(gray, p, verbose_level);
    savePyramid("pyramid/02_octave", octaves);

    std::vector<Octave> dog = buildDoG(octaves, p, verbose_level);
    savePyramid("pyramidDoG/03_dog_octave", dog, true);

    kpts = findScaleSpaceExtrema(dog, p, verbose_level);
    // ориентация ключевых точек это довольно дорогая операция
    // в случае если пользователь просит малое количество лучших точек (например, 1000, а без порога нашлось 20000),
    // то по производительности очень оправдано сразу их здесь и выбрать, чтобы не тащить до самого конца где все равно отбросим
    kpts = selectTopKeypoints(kpts, p, verbose_level);

    kpts = computeOrientations(kpts, octaves, p, verbose_level);
    // после подсчета ориентаций количество могло возрасти (и скорее всего возросло)
    // нужно снова выбрать лучшие точки чтобы уложиться в бюджет
    kpts = selectTopKeypoints(kpts, p, verbose_level);

    if (verbose_level >= 2) {
        cv::Mat kpts_img;
        cv::drawKeypoints(img, kpts, kpts_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        saveImg("04_keypoints.jpg", kpts_img);
    }

    std::tie(desc, kpts) = computeDescriptors(kpts, octaves, p, verbose_level);

    //  всегда ли мы получаем ровно столько точек сколько запросили в параметре nfeatures? в каких случаях это не так и в какую сторону?
    //   как подкрутить алгоритм, чтобы всегда выдавать ровно запрошенное количество точек (когда это в принципе возможно) но не сильно просесть в производительности?
    // алгоритм может выдать меньше точек, чем запрошено если изначально их задетектилось меньше,
    // чтобы это пофиксить можно, например, увеличивать количество слоев в октаве пока точек не станет достаточно много(но это неэффективно),
    // также можно динамически изменять порог threshold в процессе детектинга точек, и в зависимости от их количества увеличивать или уменьшать его

}

void phg::SIFT::saveImg(const std::string& name, const cv::Mat& img) const
{
    if (verbose_level < 2 || debug_folder.empty()) {
        return;
    }

    cv::Mat out;
    if (img.depth() == CV_32F) {
        img.convertTo(out, CV_8U, 255.0);
    } else {
        out = img;
    }
    cv::imwrite(debug_folder + name, out);
}

void phg::SIFT::savePyramid(const std::string& name, const std::vector<Octave>& pyramid, bool normalize) const
{
    if (verbose_level < 2 || debug_folder.empty()) {
        return;
    }

    cv::Size size = pyramid.front().layers.front().size();

    for (size_t o = 0; o < pyramid.size(); ++o) {
        std::cout << "saving octave " << o << std::endl;

        const Octave& octave = pyramid[o];

        for (size_t i = 0; i < octave.layers.size(); ++i) {
            cv::Mat layer = octave.layers[i].clone();

            cv::resize(layer, layer, size, 0, 0, cv::INTER_LINEAR);

            if (normalize) {
                double mn, mx;
                cv::minMaxLoc(layer, &mn, &mx);
                if (mx - mn > 1e-8) {
                    layer = (layer - mn) / (mx - mn);
                } else {
                    layer.setTo(0.5);
                }
            }

            std::stringstream ss;
            ss << name << "_" << o << "_layer_" << i << ".png";
            saveImg(ss.str(), layer);
        }
    }
}
