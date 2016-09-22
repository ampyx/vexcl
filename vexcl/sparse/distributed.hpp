#ifndef VEXCL_SPARSE_DISTRIBUTED_HPP
#define VEXCL_SPARSE_DISTRIBUTED_HPP

namespace vex {
namespace sparse {

template <class Matrix>
class distributed {
    public:
        typedef typename Matrix::value_type value_type;

        template <class PtrRange, class ColRange, class ValRange>
        distributed(
                const std::vector<backend::command_queue> &q,
                size_t nrows, size_t ncols,
                const PtrRange &ptr,
                const ColRange &col,
                const ValRange &val
           )
            : q(q), n(nrows), m(ncols), nnz(boost::size(val)),
              Aloc(q.size()), Arem(q.size())
        {
            typedef typename boost::range_value< typename std::decay<PtrRange>::type >::type ptr_type;
            typedef typename boost::range_value< typename std::decay<ColRange>::type >::type col_type;
            typedef typename boost::range_value< typename std::decay<ValRange>::type >::type val_type;

            if (q.size() == 1) {
                Aloc[0] = std::make_shared<Matrix>(q, ptr, col, val);
                return;
            }

            std::vector<size_t> row_part = partition(n, q);
            std::vector<size_t> col_part = partition(m, q);

            for(size_t d = 0; d < q.size(); ++d) {
                size_t loc_rows = row_part[d+1] - row_part[d];
                size_t loc_beg  = col_part[d];
                size_t loc_end  = col_part[d+1];

                std::vector<ptr_type> loc_ptr(loc_rows + 1); loc_ptr[0] = 0;
                std::vector<ptr_type> rem_ptr(loc_rows + 1); rem_ptr[0] = 0;

                // Count nonzeros in local and remote parts of the matrix.
                for(size_t i = row_part[d], ii = 0; i < row_part[d+1]; ++i, ++ii) {
                    loc_ptr[ii+1] = loc_ptr[ii];
                    rem_ptr[ii+1] = rem_ptr[ii];

                    for(ptr_type j = ptr[i]; j < ptr[i+1]; ++j) {
                        col_type c = col[j];

                        if (loc_beg <= c && c < loc_end) {
                            ++loc_ptr[ii+1];
                        } else {
                            ++rem_ptr[ii+1];
                        }
                    }
                }

                // Fill local and remote parts of the matrix.
                std::vector<col_type> loc_col; loc_col.reserve(loc_ptr.back());
                std::vector<val_type> loc_val; loc_val.reserve(loc_ptr.back());

                std::vector<col_type> rem_col; rem_col.reserve(rem_ptr.back());
                std::vector<val_type> rem_val; rem_val.reserve(rem_ptr.back());

                for(size_t i = row_part[d], ii = 0; i < row_part[d+1]; ++i, ++ii) {
                    for(ptr_type j = ptr[i]; j < ptr[i+1]; ++j) {
                        col_type c = col[j];
                        val_type v = val[j];

                        if (loc_beg <= c && c < loc_end) {
                            loc_col.push_back(c - loc_beg);
                            loc_val.push_back(v);
                        } else {
                            rem_col.push_back(c);
                            rem_val.push_back(v);
                        }
                    }
                }

                // Get list of unique remote columns.
                std::vector<col_type> rcols = rem_col;
                std::sort(rcols.begin(), rcols.end());
                rcols.erase(std::unique(rcols.begin(), rcols.end()), rcols.end());

                // Renumber remote columns.
                col_type nrcols = rcols.size();
                std::unordered_map<col_type, col_type> idx(2 * nrcols);

                for(col_type i = 0; i < nrcols; ++i) {
                    idx.insert(idx.end(), std::make_pair(rcols[i], i));
                }
            }
        }
    private:
        size_t n, m, nnz;
        std::vector<std::shared_ptr<Matrix>> Aloc, Arem;
};

} // namespace sparse
} // namespace vex

#endif
