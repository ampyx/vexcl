#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>
#include <functional>
#include <numeric>
#include <cmath>

//#define VEXCL_SHOW_KERNELS
#include <vexcl/vexcl.hpp>

using namespace vex;

bool run_test(const std::string &name, std::function<bool()> test) {
    char fc = std::cout.fill('.');
    std::cout << name << ": " << std::setw(62 - name.size()) << "." << std::flush;
    std::cout.fill(fc);

    bool rc = test();
    std::cout << (rc ? " success." : " failed.") << std::endl;
    return rc;
}

extern const char chk_if_gr_body[] = "return prm1 > prm2 ? 1 : 0;";

int main() {
    srand(time(0));

    try {
	vex::Context ctx(Filter::DoublePrecision && Filter::Env);
	std::cout << ctx << std::endl;

	if (ctx.queue().empty()) {
	    std::cerr << "No OpenCL devices found." << std::endl;
	    return 1;
	}

	run_test("Empty vector construction", [&]() {
		bool rc = true;
		vex::vector<double> x;
		rc = rc && (x.size() == 0);
		rc = rc && (x.end() - x.begin() == 0);
		return rc;
		});

	run_test("Vector construction from size", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(ctx.queue(), N);
		rc = rc && (x.size() == N);
		rc = rc && (x.end() == x.begin() + N);
		return rc;
		});

	run_test("Vector construction from std::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		std::vector<double> y(N);
		vex::vector<double> X(ctx.queue(), x);
		rc = rc && (X.size() == x.size());
		rc = rc && (X.end() == X.begin() + x.size());
		copy(X, y);
		std::transform(x.begin(), x.end(), y.begin(), y.begin(),
		    [](double a, double b) { return a - b; });
		rc = rc && std::all_of(y.begin(), y.end(),
		    [](double a) {return a == 0; });
		return rc;
		});

	run_test("Vector construction from size and host pointer", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		std::vector<double> y(N);
		vex::vector<double> X(ctx.queue(), N, x.data());
		rc = rc && (X.size() == x.size());
		rc = rc && (X.end() == X.begin() + x.size());
		copy(X, y);
		std::transform(x.begin(), x.end(), y.begin(), y.begin(),
		    [](double a, double b) { return a - b; });
		rc = rc && std::all_of(y.begin(), y.end(),
		    [](double a) {return a == 0; });
		return rc;
		});

	run_test("Vector copy construction", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x1;
		vex::vector<double> x2(x1);

		vex::vector<double> y1(ctx.queue(), N);
		vex::vector<double> y2(y1);
		rc = rc && (x1.size() == x2.size() && x1.size() == 0);
		rc = rc && (y1.size() == y2.size() && y1.size() == N);
		return rc;
		});

	run_test("Vector move construction from vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(ctx.queue(), N);
		x = 42;
		vex::vector<double> y = std::move(x);
		rc = rc && (y.size() == N);
		rc = rc && (y.end() == y.begin() + N);
		Reductor<double,MIN> min(ctx.queue());
		Reductor<double,MAX> max(ctx.queue());
		rc = rc && min(y) == 42;
		rc = rc && max(y) == 42;
		return rc;
		});

	run_test("Vector move assignment", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		vex::vector<double> X(ctx.queue(), x);
		vex::vector<double> Y = std::move(X);
		rc = rc && (Y.size() == x.size());
		Reductor<size_t,SUM> sum(ctx.queue());
		rc = rc && sum(Y != x[0]) == 0;
		return rc;
		});

	run_test("Vector swap", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(ctx.queue(), N);
		vex::vector<double> y(ctx.queue(), N/2);
		x = 42;
		y = 67;
		swap(x, y);
		rc = rc && (y.size() == N);
		rc = rc && (x.size() == N/2);
		Reductor<size_t,SUM> sum(ctx.queue());
		rc = rc && sum(y != 42) == 0;
		rc = rc && sum(x != 67) == 0;
		return rc;
		});

	run_test("Vector resize from std::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		vex::vector<double> X;
		X.resize(ctx.queue(), x);
		rc = rc && (X.size() == x.size());
		Reductor<size_t,SUM> sum(ctx.queue());
		rc = rc && sum(X != 42) == 0;
		return rc;
		});

	run_test("Vector resize vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(ctx.queue(), N);
		x = 42;
		vex::vector<double> y;
		y.resize(x);
		rc = rc && (y.size() == x.size());
		Reductor<size_t,SUM> sum(ctx.queue());
		rc = rc && sum(x != y) == 0;
		return rc;
		});

	run_test("Iterate over vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(ctx.queue(), N);
		x = 42;
		rc = rc && std::all_of(x.begin(), x.end(),
		    [](double a) { return a == 42; });
		return rc;
		});

	run_test("Access vex::vector elements", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(ctx.queue(), N);
		for(uint i = 0; i < N; i++)
		    x[i] = 42;
		for(uint i = 0; i < N; i++)
		    rc == rc && (x[i] == 42);
		return rc;
		});

	run_test("Copy vex::vector to std::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N);
		vex::vector<double> X(ctx.queue(), N);
		X = 42;
		copy(X, x);
		rc = rc && std::all_of(x.begin(), x.end(),
		    [](double a) { return a == 42; });
		X = 67;
		vex::copy(X.begin(), X.end(), x.begin());
		rc = rc && std::all_of(x.begin(), x.end(),
		    [](double a) { return a == 67; });
		return rc;
		});

	run_test("Copy std::vector to vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		vex::vector<double> X(ctx.queue(), N);
		copy(x, X);
		Reductor<size_t,SUM> sum(ctx.queue());
		rc = rc && sum(X != 42) == 0;
		std::fill(x.begin(), x.end(), 67);
		vex::copy(x.begin(), x.end(), X.begin());
		rc = rc && sum(X != 67) == 0;
		return rc;
		});

	run_test("Assign expression to vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(ctx.queue(), N);
		vex::vector<double> y(ctx.queue(), N);
		vex::vector<double> z(ctx.queue(), N);
		y = 42;
		z = 67;
		x = 5 * sin(y) + z;
		Reductor<double,MAX> max(ctx.queue());
		rc = rc && max(fabs(x - (5 * sin(42) + 67))) < 1e-12;
		return rc;
		});


	run_test("Reduction", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N);
		std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });
		vex::vector<double> X(ctx.queue(), x);
		Reductor<double,SUM> sum(ctx.queue());
		Reductor<double,MIN> min(ctx.queue());
		Reductor<double,MAX> max(ctx.queue());
		rc = rc && fabs(sum(X) - std::accumulate(x.begin(), x.end(), 0.0)) < 1e-6;
		rc = rc && fabs(min(X) - *std::min_element(x.begin(), x.end())) < 1e-6;
		rc = rc && fabs(max(X) - *std::max_element(x.begin(), x.end())) < 1e-6;
		return rc;
		});


	run_test("Sparse matrix-vector product", [&]() {
		bool rc = true;
		const size_t n   = 32;
		const double h2i = (n - 1) * (n - 1);

		std::vector<size_t> row;
		std::vector<size_t> col;
		std::vector<double> val;

		row.reserve(n * n * n + 1);
		col.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);
		val.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);

		row.push_back(0);
		for(size_t k = 0, idx = 0; k < n; k++) {
		    for(size_t j = 0; j < n; j++) {
			for(size_t i = 0; i < n; i++, idx++) {
			    if (
				i == 0 || i == (n - 1) ||
				j == 0 || j == (n - 1) ||
				k == 0 || k == (n - 1)
			       )
			    {
				col.push_back(idx);
				val.push_back(1);
				row.push_back(row.back() + 1);
			    } else {
				col.push_back(idx - n * n);
				val.push_back(-h2i);

				col.push_back(idx - n);
				val.push_back(-h2i);

				col.push_back(idx - 1);
				val.push_back(-h2i);

				col.push_back(idx);
				val.push_back(6 * h2i);

				col.push_back(idx + 1);
				val.push_back(-h2i);

				col.push_back(idx + n);
				val.push_back(-h2i);

				col.push_back(idx + n * n);
				val.push_back(-h2i);

				row.push_back(row.back() + 7);
			    }
			}
		    }
		}

		std::vector<double> x(n * n * n);
		std::vector<double> y(n * n * n);
		std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });

		vex::SpMat <double> A(ctx.queue(), x.size(), row.data(), col.data(), val.data());
		vex::vector<double> X(ctx.queue(), x);
		vex::vector<double> Y(ctx.queue(), x.size());

		Y = A * X;
		copy(Y, y);

		double res = 0;
		for(size_t i = 0; i < x.size(); i++) {
		    double sum = 0;
		    for(size_t j = row[i]; j < row[i + 1]; j++)
			sum += val[j] * x[col[j]];
		    res = std::max(res, fabs(sum - y[i]));
		}

		rc = rc && res < 1e-8;

		Y = X + A * X;
		copy(Y, y);

		res = 0;
		for(size_t i = 0; i < x.size(); i++) {
		    double sum = 0;
		    for(size_t j = row[i]; j < row[i + 1]; j++)
			sum += val[j] * x[col[j]];
		    res = std::max(res, fabs(sum + x[i] - y[i]));
		}

		rc = rc && res < 1e-8;

		return rc;
	});

	run_test("Sparse matrix-vector product (CCSR format)", [&]() {
		bool rc = true;
		const uint n   = 32;
		const double h2i = (n - 1) * (n - 1);

		std::vector<size_t> idx;
		std::vector<size_t> row(3);
		std::vector<int>    col(8);
		std::vector<double> val(8);

		idx.reserve(n * n * n);

		row[0] = 0;
		row[1] = 1;
		row[2] = 8;

		col[0] = 0;
		val[0] = 1;

		col[1] = -(n * n);
		col[2] =    -n;
		col[3] =    -1;
		col[4] =     0;
		col[5] =     1;
		col[6] =     n;
		col[7] =  (n * n);

		val[1] = -h2i;
		val[2] = -h2i;
		val[3] = -h2i;
		val[4] =  h2i * 6;
		val[5] = -h2i;
		val[6] = -h2i;
		val[7] = -h2i;

		for(size_t k = 0; k < n; k++) {
		    for(size_t j = 0; j < n; j++) {
			for(size_t i = 0; i < n; i++) {
			    if (
				i == 0 || i == (n - 1) ||
				j == 0 || j == (n - 1) ||
				k == 0 || k == (n - 1)
			       )
			    {
				idx.push_back(0);
			    } else {
				idx.push_back(1);
			    }
			}
		    }
		}

		std::vector<double> x(n * n * n);
		std::vector<double> y(n * n * n);
		std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });

		std::vector<cl::CommandQueue> q1(1, ctx.queue()[0]);

		vex::SpMatCCSR<double,int> A(q1[0], x.size(), row.size() - 1,
			idx.data(), row.data(), col.data(), val.data());

		vex::vector<double> X(q1, x);
		vex::vector<double> Y(q1, x.size());

		Y = A * X;
		copy(Y, y);

		double res = 0;
		for(size_t i = 0; i < x.size(); i++) {
		    double sum = 0;
		    for(size_t j = row[idx[i]]; j < row[idx[i] + 1]; j++)
			sum += val[j] * x[i + col[j]];
		    res = std::max(res, fabs(sum - y[i]));
		}

		rc = rc && res < 1e-8;

		Y = X + A * X;
		copy(Y, y);

		res = 0;
		for(size_t i = 0; i < x.size(); i++) {
		    double sum = 0;
		    for(size_t j = row[idx[i]]; j < row[idx[i] + 1]; j++)
			sum += val[j] * x[i + col[j]];
		    res = std::max(res, fabs(sum + x[i] - y[i]));
		}

		rc = rc && res < 1e-8;

		return rc;
	});

	run_test("Builtin function with one argument", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N);
		std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });
		vex::vector<double> X(ctx.queue(), x);
		Reductor<double,SUM> sum(ctx.queue());
		rc = rc && 1e-8 > fabs(sum(sin(X)) -
		    std::accumulate(x.begin(), x.end(), 0.0, [](double s, double v) {
			return s + sin(v);
			}));
		rc = rc && 1e-8 > fabs(sum(cos(X)) -
		    std::accumulate(x.begin(), x.end(), 0.0, [](double s, double v) {
			return s + cos(v);
			}));
		return rc;
		});

#ifdef VEXCL_VARIADIC_TEMPLATES
	run_test("Builtin function with two arguments", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N);
		std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });
		vex::vector<double> X(ctx.queue(), x);
		Reductor<double,SUM> sum(ctx.queue());
		rc = rc && 1e-8 > fabs(sum(pow(X, 2.0)) -
		    std::accumulate(x.begin(), x.end(), 0.0, [](double s, double v) {
			return s + pow(v, 2.0);
			}));
		return rc;
		});
#endif

	run_test("Custom function", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(ctx.queue(), N);
		vex::vector<double> y(ctx.queue(), N);
		x = 1;
		y = 2;
		UserFunction<chk_if_gr_body, size_t(double, double)> chk_if_greater;
		Reductor<size_t,SUM> sum(ctx.queue());
		rc = rc && sum(chk_if_greater(x, y)) == 0;
		rc = rc && sum(chk_if_greater(y, x)) == N;
		rc = rc && sum(x > y) == 0;
		rc = rc && sum(x < y) == N;
		return rc;
		});

	run_test("Access multivector's elements, copy data", [&]() {
		bool rc = true;
		const size_t n = 1024;
		const size_t m = 4;
		std::vector<float> host(n * m);
		std::generate(host.begin(), host.end(),
		    [](){ return (float)rand() / RAND_MAX; });
		multivector<float, m> x(ctx.queue(), n);
		copy(host, x);
		for(size_t i = 0; i < n; i++) {
		    std::array<float,m> val = x[i];
		    for(uint j = 0; j < m; j++) {
			rc = rc && val[j] == host[j * n + i];
			val[j] = 0;
		    }
		    x[i] = val;
		}
		copy(x, host);
		rc = rc && 0 == *std::min_element(host.begin(), host.end());
		rc = rc && 0 == *std::max_element(host.begin(), host.end());
		return rc;
	});

	run_test("Simple arithmetic with multivectors", [&]() {
		bool rc = true;
		const size_t n = 1024;
		const size_t m = 4;
		std::vector<float> host(n * m);
		std::generate(host.begin(), host.end(),
		    [](){ return (float)rand() / RAND_MAX; });
		multivector<float, m> x(ctx.queue(), n);
		multivector<float, m> y(ctx.queue(), host);
		multivector<float, m> z(ctx.queue(), host);
		Reductor<float,MIN> min(ctx.queue());
		Reductor<float,MAX> max(ctx.queue());

		std::array<int, m> v;
		for(uint i = 0; i < m; i++) v[i] = i;
		x = v;
		std::array<float, m> xmin = min(x);
		std::array<float, m> xmax = max(x);
		for(uint i = 0; i < m; i++) {
		    rc = rc && xmin[i] == v[i];
		    rc = rc && xmax[i] == v[i];
		}

		x = 2 * y + z;
		std::transform(host.begin(), host.end(), host.begin(), [](float x) {
		    return 2 * x + x;
		    });
		for(uint i = 0; i < m; i++) {
		    rc = rc && min(x(i)) == *min_element(
			host.begin() + i * n, host.begin() + (i + 1) * n);
		    rc = rc && max(x(i)) == *max_element(
			host.begin() + i * n, host.begin() + (i + 1) * n);
		}
		return rc;
	});

	run_test("One-argument builtin function call for multivector", [&]() {
		bool rc = true;
		const size_t n = 1024;
		const size_t m = 4;
		std::vector<double> host(n * m);
		std::generate(host.begin(), host.end(),
		    [](){ return (double)rand() / RAND_MAX; });
		multivector<double, m> x(ctx.queue(), n);
		multivector<double, m> y(ctx.queue(), host);
		x = cos(y);
		for(size_t k = 0; k < 10; k++) {
		    size_t i = rand() % n;
		    std::array<double,m> val = x[i];
		    for(uint j = 0; j < m; j++)
			rc = rc && fabs(val[j] - cos(host[j * n + i])) < 1e-8;
		}
		return rc;
	});

	run_test("Two-arguments builtin function call for multivector", [&]() {
		bool rc = true;
		const size_t n = 1024;
		const size_t m = 4;
		std::vector<double> host(n * m);
		std::generate(host.begin(), host.end(),
		    [](){ return (double)rand() / RAND_MAX; });
		multivector<double, m> x(ctx.queue(), n);
		multivector<double, m> y(ctx.queue(), host);
		x = pow(y, 2.0);
		for(size_t k = 0; k < 10; k++) {
		    size_t i = rand() % n;
		    std::array<double,m> val = x[i];
		    for(uint j = 0; j < m; j++)
			rc = rc && fabs(val[j] - pow(host[j * n + i], 2.0)) < 1e-8;
		}
		return rc;
	});

	run_test("Custom function for multivector", [&]() {
		bool rc = true;
		const size_t n = 1024;
		const size_t m = 4;
		multivector<double, m> x(ctx.queue(), n);
		multivector<double, m> y(ctx.queue(), n);
		x = 1;
		y = 2;
		UserFunction<chk_if_gr_body, size_t(double, double)> chk_if_greater;
		x = chk_if_greater(x, y);
		for(size_t k = 0; k < 10; k++) {
		    size_t i = rand() % n;
		    std::array<double,m> val = x[i];
		    for(uint j = 0; j < m; j++)
			rc = rc && val[j] == 0;
		}
		return rc;
		});


	run_test("Reduction of multivector", [&]() {
		bool rc = true;
		const size_t n = 1024;
		const size_t m = 4;
		std::vector<double> host(n * m);
		std::generate(host.begin(), host.end(),
		    [](){ return (double)rand() / RAND_MAX; });
		multivector<double, m> x(ctx.queue(), host);
		Reductor<double,SUM> sum(ctx.queue());
		std::array<double,m> s = sum(x);
		for(uint i = 0; i < m; i++) {
		    rc = rc && fabs(
			s[i] - std::accumulate(
			    host.begin() + i * n, host.begin() + (i + 1) * n, 0.0)
			) < 1e-6;
		}
		return rc;
		});

	run_test("Sparse matrix-multivector product", [&]() {
		bool rc = true;
		const size_t n   = 32;
		const size_t N   = n * n * n;
		const size_t m   = 2;
		const double h2i = (n - 1) * (n - 1);

		std::vector<size_t> row;
		std::vector<size_t> col;
		std::vector<double> val;

		row.reserve(N + 1);
		col.reserve(6 * (n - 2) * (n - 2) * (n - 2) + N);
		val.reserve(6 * (n - 2) * (n - 2) * (n - 2) + N);

		row.push_back(0);
		for(size_t k = 0, idx = 0; k < n; k++) {
		    for(size_t j = 0; j < n; j++) {
			for(size_t i = 0; i < n; i++, idx++) {
			    if (
				i == 0 || i == (n - 1) ||
				j == 0 || j == (n - 1) ||
				k == 0 || k == (n - 1)
			       )
			    {
				col.push_back(idx);
				val.push_back(1);
				row.push_back(row.back() + 1);
			    } else {
				col.push_back(idx - n * n);
				val.push_back(-h2i);

				col.push_back(idx - n);
				val.push_back(-h2i);

				col.push_back(idx - 1);
				val.push_back(-h2i);

				col.push_back(idx);
				val.push_back(6 * h2i);

				col.push_back(idx + 1);
				val.push_back(-h2i);

				col.push_back(idx + n);
				val.push_back(-h2i);

				col.push_back(idx + n * n);
				val.push_back(-h2i);

				row.push_back(row.back() + 7);
			    }
			}
		    }
		}

		std::vector<double> x(N * m);
		std::vector<double> y(N * m);
		std::generate(x.begin(), x.end(), []() {
			return (double)rand() / RAND_MAX;
			});

		vex::SpMat <double> A(ctx.queue(), N,
			row.data(), col.data(), val.data());

		vex::multivector<double,m> X(ctx.queue(), x);
		vex::multivector<double,m> Y(ctx.queue(), N);

		Y = A * X;
		copy(Y, y);

		double res = 0;
		for(uint k = 0; k < m; k++)
		    for(size_t i = 0; i < N; i++) {
			double sum = 0;
			for(size_t j = row[i]; j < row[i + 1]; j++)
			    sum += val[j] * x[col[j] + k * N];
			res = std::max(res, fabs(sum - y[i + k * N]));
		    }

		rc = rc && res < 1e-8;

		Y = X + A * X;
		copy(Y, y);

		res = 0;
		for(uint k = 0; k < m; k++)
		    for(size_t i = 0; i < N; i++) {
			double sum = 0;
			for(size_t j = row[i]; j < row[i + 1]; j++)
			    sum += val[j] * x[col[j] + k * N];
			res = std::max(res, fabs(sum + x[i + k * N] - y[i + k * N]));
		    }

		rc = rc && res < 1e-8;

		return rc;
	});

	run_test("Sparse matrix-multivector product (CCSR format)", [&]() {
		bool rc = true;
		const uint n     = 32;
		const uint N     = n * n * n;
		const uint m     = 2;
		const double h2i = (n - 1) * (n - 1);

		std::vector<size_t> idx;
		std::vector<size_t> row(3);
		std::vector<int>    col(8);
		std::vector<double> val(8);

		idx.reserve(N);

		row[0] = 0;
		row[1] = 1;
		row[2] = 8;

		col[0] = 0;
		val[0] = 1;

		col[1] = -(n * n);
		col[2] =    -n;
		col[3] =    -1;
		col[4] =     0;
		col[5] =     1;
		col[6] =     n;
		col[7] =  (n * n);

		val[1] = -h2i;
		val[2] = -h2i;
		val[3] = -h2i;
		val[4] =  h2i * 6;
		val[5] = -h2i;
		val[6] = -h2i;
		val[7] = -h2i;

		for(size_t k = 0; k < n; k++) {
		    for(size_t j = 0; j < n; j++) {
			for(size_t i = 0; i < n; i++) {
			    if (
				i == 0 || i == (n - 1) ||
				j == 0 || j == (n - 1) ||
				k == 0 || k == (n - 1)
			       )
			    {
				idx.push_back(0);
			    } else {
				idx.push_back(1);
			    }
			}
		    }
		}

		std::vector<double> x(N * m);
		std::vector<double> y(N * m);
		std::generate(x.begin(), x.end(), []() {
			return (double)rand() / RAND_MAX;
			});

		std::vector<cl::CommandQueue> q1(1, ctx.queue()[0]);

		vex::SpMatCCSR<double,int> A(q1[0], N, row.size() - 1,
			idx.data(), row.data(), col.data(), val.data());

		vex::multivector<double,m> X(q1, x);
		vex::multivector<double,m> Y(q1, N);

		Y = A * X;
		copy(Y, y);

		double res = 0;
		for(uint k = 0; k < m; k++)
		    for(size_t i = 0; i < N; i++) {
			double sum = 0;
			for(size_t j = row[idx[i]]; j < row[idx[i] + 1]; j++)
			    sum += val[j] * x[i + col[j] + k * N];
			res = std::max(res, fabs(sum - y[i + k * N]));
		    }

		rc = rc && res < 1e-8;

		Y = X + A * X;
		copy(Y, y);

		res = 0;
		for(uint k = 0; k < m; k++)
		    for(size_t i = 0; i < N; i++) {
			double sum = 0;
			for(size_t j = row[idx[i]]; j < row[idx[i] + 1]; j++)
			    sum += val[j] * x[i + col[j] + k * N];
			res = std::max(res, fabs(sum + x[i + k * N] - y[i + k * N]));
		    }

		rc = rc && res < 1e-8;

		return rc;
	});


    } catch (const cl::Error &err) {
	std::cerr << "OpenCL error: " << err << std::endl;
	return 1;
    } catch (const std::exception &err) {
	std::cerr << "Error: " << err.what() << std::endl;
	return 1;
    } catch (...) {
	std::cerr << "Unknown error" << std::endl;
	return 1;
    }

    return 0;
}