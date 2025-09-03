import numpy as np
class MatlabFuncHelper(object):
    # configurations
    BATCH_SIZE_NO = None;
    nan = np.nan;
    NaN = np.nan;
    NAN = np.nan;
    # settings
    batch_size = BATCH_SIZE_NO;
    
    ###########################################################################
    # Batch
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size;
    def getBatchSize(self):
        return self.batch_size;
    
    ###########################################################################   
    '''
    check input is a vector like [(batch_size), n],  [(batch_size), ..., n, 1] or [(batch_size), ..., 1, n] 
    '''
    def isvector(self, mat):
        mat = np.asarray(mat);
        if self.batch_size == self.BATCH_SIZE_NO:
            return mat.ndim == 1 or mat.ndim >= 2 and (mat.shape[-2] == 1 or mat.shape[-1] == 1);
        else:
            if mat.shape[0] != self.batch_size:
                raise Exception("The input does not has the required batch size.");
            else:
                return mat.ndim == 2 or mat.ndim >= 3 and (mat.shape[-2] == 1 or mat.shape[-1] == 1);
            
    '''
    check input is a matrix like [(batch_size), ..., n, m]
    '''
    def ismatrix(self, mat):
        mat = np.asarray(mat);
        if self.batch_size == self.BATCH_SIZE_NO:
            return mat.ndim == 2 and mat.shape[-2] > 1 and mat.shape[-1] > 1;
        else:
            if mat.shape[0] != self.batch_size:
                raise Exception("The input does not has the required batch size.");
            else:
                return mat.ndim >= 3 and mat.shape[-2] > 1 and mat.shape[-1] > 1;
    
    '''
    is NaN
    '''
    def isnan(self, mat):
        return np.isnan(mat);
    
    ###########################################################################
    # generators
    '''
    generate NaN
    '''
    def nan(self, *args):
        if len(args) == 0:
            if self.batch_size == self.BATCH_SIZE_NO:
                out = np.nan;
            else:
                out = np.tile(np.nan, (self.batch_size));
        else:
            shape = list(args);
            if self.batch_size != self.BATCH_SIZE_NO:
                shape.insert(0, self.batch_size);
            out = np.tile(np.nan, shape);
        return out;
    
    '''
    generate a matrix of all zeros
    @in1:   1st dimension
    @in...: other dimensions
    @order: 'C': only consider given dimensions, 'F': if only given 1 dimension, this is for the row and column
    '''
    def zeros(self, in1, *args, order='C'):
        shape = self._shape_calc_(in1, *args, order=order);
        if self.batch_size != self.BATCH_SIZE_NO:
            shape.insert(0, self.batch_size);
        return np.zeros(shape);    
    
    '''
    generate a matrix full of ones
    @in1:   1st dimension
    @in...: other dimensions
    @order: 'C': only consider given dimensions, 'F': if only given 1 dimension, this is for the row and column
    '''
    def ones(self, in1, *args, order='C'):
        shape = self._shape_calc_(in1, *args, order=order);
        if self.batch_size != self.BATCH_SIZE_NO:
            shape.insert(0, self.batch_size);
        return np.ones(shape);
    
    '''
    return an identity matrix
    @in1:   1st dimension
    @in...: other dimensions
    '''
    def eye(self, in1, *args):
        in2 = in1 if len(args) == 0 else args[0];
        out = np.eye(in1, in2);
        if self.batch_size !=self.BATCH_SIZE_NO:
            out = np.tile(out,(self.batch_size, 1, 1));
        return out;
    
    '''
    generate a sequence
    @in1:   (1) beg  (2) beg (3) end
    @in2:   (1) step (2) end
    @in3:   (1) end
    @order: 'C', (1,2,3)output uses `end-1`, (3)`beg`=0; 'F', (1,2,3)output uses `end`, (3)`beg`=1
    '''
    def seq(self, in1, *args, order='C'):
        order = order.upper();
        beg = 0;
        step = 1;
        end = 0;
        if len(args) == 0:
            end = in1;
            if order == 'F':
                beg = 1;
        elif len(args) == 1:
            beg = in1;
            end = args[0];
            if order == 'F':
                end = end + 1;
        elif len(args) == 2:
            beg = in1;
            step = args[0];
            end = args[1];
            if order == 'F':
                end = end + 1;
        out = np.arange(beg, end, step);
        if self.batch_size != self.BATCH_SIZE_NO:
            out = np.tile(out,(self.batch_size, 1));
        return out;
    
    '''
    generate random values from [0, 1) following a uniform distribution
    @in1:   1st dimension
    @in...: other dimensions
    '''
    def rand(self, in1, *args):
        if self.batch_size == self.BATCH_SIZE_NO:
            return np.random.rand(in1, *args);
        else:
            return np.random.rand(self.batch_size, in1, *args);
        
    '''
    generate random values following the standard normal distribution 
    @in1:   1st dimension
    @in...: other dimensions
    '''
    def randn(self, in1, *args):
        if self.batch_size == self.BATCH_SIZE_NO:
            return np.random.randn(in1, *args);
        else:
            return np.random.randn(self.batch_size, in1, *args);
    
    '''
    generate the DFT matrix
    @in1:   1st dimension
    '''
    def dftmtx(self, in1):
        return np.fft.fft(self.eye(in1))/np.sqrt(in1);
    
    ###########################################################################
    # transformers
    '''
    remove redundant dimension except for the batch_size
    '''
    def squeeze(self, mat):
        mat = np.asarray(mat);
        out = np.squeeze(mat);
        if self.batch_size == 1 and mat.ndim > 0:
            if mat.shape[0] == self.batch_size:
                out = np.expand_dims(out, 0);
        return out;
    
    '''
    reshape
    @in1:   1st dimension
    @in...: other dimensions
    @order: 'C': only consider given dimensions, 'F': if only given 1 dimension, this is for the row and column
    '''
    def reshape(self, mat, in1, *args, order='C'):
        shape = list(args);
        shape.insert(0, in1);
        if self.batch_size != self.BATCH_SIZE_NO:
            shape.insert(0, self.batch_size);
        return np.reshape(mat, shape, order=order);
    
    '''
    repeat the matrix in the given dimension
    @in1:   1st dimension
    @in...: other dimensions
    @order: 'C': only consider given dimensions, 'F': if only given 1 dimension, this is for the row and column
    '''
    # repeat the matrix in the given dimension (the batch dim is repeated as 1)
    def repmat1(self, mat, in1, *args, order='C'):
        mat = np.asarray(mat);
        mat_dims = mat.ndim;
        shape_ori_len, shape = self.rempat_build_shape(mat_dims, in1, *args, order=order);
        # batch size
        if self.batch_size != self.BATCH_SIZE_NO:
            # the given shape has a smaller dimension than that figure of the given matrix
            # the (batch_size) is ignored 
            if shape_ori_len <= mat_dims:    
                shape.insert(0, 1);
        return np.tile(mat, shape);
    # repeat the matrix in the given dimension (the batch dim is repeated as batch_size)
    def repmatN(self, mat, in1, *args, order='C'):
        mat = np.asarray(mat);
        mat_dims = mat.ndim;
        shape_ori_len, shape = self.rempat_build_shape(mat_dims, in1, *args, order=order);
        # batch size
        if self.batch_size != self.BATCH_SIZE_NO:
            # the given shape has a smaller dimension than that figure of the given matrix
            # the (batch_size) is ignored 
            if shape_ori_len <= mat_dims:    
                shape.insert(0, self.batch_size);
        return np.tile(mat, shape);
    # calculate [shape_len, shape] for repmat before batch_size
    def rempat_build_shape(self, mat_dims, in1, *args, order='C'):
        shape = self._shape_calc_(in1, *args, order=order);
        # align
        shape_ori_len = len(shape); # original shape length
        if shape_ori_len < mat_dims:
            shape = [1]*(mat_dims - shape_ori_len) + shape;
        return shape_ori_len, shape;
    
    '''
    generate a matrix based on its diag or get a diagonal matrix from its vector
    @mat:   a vector as [(batch_size), n, 1] or [(batch_size), 1, n]; if n == 1, 
            it will be taken as [(batch_size), n, 1].
            Or a square matrix [(batch_size), n, n]
    '''
    def diag(self, mat):
        # input check
        if self.batch_size == self.BATCH_SIZE_NO and mat.ndim > 2 or self.batch_size != self.BATCH_SIZE_NO and mat.ndim > 3:
            raise Exception("The input dimension is over 2D, so we cannot know you want to take as a diagonal or build a matrix based on its diagonal.");
        # remove redundant dimensions for [(batch_size), n, 1] or [(batch_size), 1, n]
        if self.batch_size == self.BATCH_SIZE_NO:
            if mat.ndim >= 2:
                if mat.shape[-1] == 1:
                    mat = mat.squeeze(-1);
                elif mat.shape[-2] == 1:
                    mat = mat.squeeze(-2);
        else:
            if mat.ndim >= 3:
                if mat.shape[-1] == 1:
                    mat = mat.squeeze(-1);
                elif mat.shape[-2] == 1:
                    mat = mat.squeeze(-2);
        # generate output
        out = None;
        # diag_vec_len = diag_vec.shape[-1];
        if self.batch_size is self.BATCH_SIZE_NO:
            out = np.diag(mat);
        else:
            # np.zeros only take real numbers by default, here we need to put complex value into it
            out = [];
            # create output
            for batch_id in range(self.batch_size):
                out.append(np.diag(mat[batch_id, ...]));
            out = np.asarray(out);
        # add extra dimension in the end
        if self.ismatrix(mat):
            out = np.expand_dims(out, axis=-1);
        return out;
    
    '''
    circular shift (1st index except for the batch size)
    @mat:   [(batch_size), dim1, dim2, ...]
    @step:  a scalar or an 1D array (only for batch)
    '''
    def circshift(self, mat, step):
        mat = np.asarray(mat);
        step = np.asarray(step);
        if self.batch_size == self.BATCH_SIZE_NO:
            out = np.roll(mat, step, axis=0);
        else:
            mat_shape = list(mat.shape);
            mat_shape.insert(0, self.batch_size);
            out = np.zeros(mat.shape, dtype=mat.dtype);
            for batch_id in range(self.batch_size):
                if step.ndim == 0:
                    out[batch_id, ...] = np.roll(mat[batch_id, ...], step, axis=0);
                else:
                    out[batch_id, ...] = np.roll(mat[batch_id, ...], step[batch_id], axis=0);
        return out;

    ###########################################################################
    # Maths
    '''
    sum
    '''
    def sum(self, mat, *, axis=-1):
        return np.sum(mat, axis=axis);
    '''
    sum (the return value has the same dimension number as the given)
    '''
    def sum1(self, mat, *, axis=-1):
        return np.expand_dims(np.sum(mat, axis=axis), axis=axis);
    
    '''
    return the maximum of a matrix or the maximum of two matrices (for complex value, we compare the magnitude)
    @in1:   the matrix to find the maximal value
    @in2:   a scalar or a matrix to be compared with in1. If not given, it means return the maximal value of in1
    @axis:  the axis to compare or find the maximal value
    '''
    def max(self, in1, *args, axis=-1):
        out = None;
        in1 = np.asarray(in1);
        in2 = None;
        if len(args) > 0:
            if isinstance(args[0], float) or isinstance(args[0], int):
                in2 = args[0];
            elif args[0].ndim == 0 or in1.shape == args[0].shape:
                in2 = args[0].astype(in1.dtype);
            else:
                raise Exception("The input two matrices must have the same shape.");
        # non-complex value
        if in1.dtype != np.csingle and in1.dtype != np.complex_:
            if len(args) == 0:
                out = in1.max(axis=axis);
            else:
                out = np.where(in1>in2, in1, in2);
        # complex value
        else:
            if len(args) == 0:
                out = np.take_along_axis(in1, abs(in1).argmax(axis=axis, keepdims=True), axis).squeeze(axis);
            else:
                out = np.where(abs(in1)>abs(in2), in1, in2);
        return out;
    
    def max1(self, in1, *args, axis=-1):
        in1 = np.asarray(in1);
        if len(args) > 0:
            return self.max(in1, *args, axis=axis);
        else:
            # non-complex value
            if in1.dtype != np.csingle and in1.dtype != np.complex_:
                return np.max(in1, axis=axis);
            else:
                return np.take_along_axis(in1, np.argmax(abs(in1), axis=axis, keepdims=True), axis);
    
    '''
    Kronecker product
    @a: a matrix like [(batch_size), ..., i, j]
    @b: a matrix like [(batch_size), ..., k, l]
    '''
    def kron(self, a, b):
        a = np.asarray(a);
        b = np.asarray(b);
        if self.batch_size == self.BATCH_SIZE_NO:
            out = np.kron(a, b);
        else:
            if a.ndim < 3:
                a = np.expand_dims(a, -2);
            if b.ndim < 3:
                b = np.expand_dims(b, -2);
            if len(a.shape) != len(b.shape):
                raise Exception("The two inputs dimensions are not same.");
            shape = list(a.shape);
            shape[-1] = a.shape[-1]*b.shape[-1];
            shape[-2] = a.shape[-1]*b.shape[-1];
            out = np.einsum('...ij,...kl->...ikjl', a, b).reshape(shape);
        return out;
        

    ###########################################################################
    # private methods
    '''
    calculate the shape based on the given dimensions
    '''
    def _shape_calc_(self, in1, *args, order='C'):
        order = order.upper();
        shape = list(args);
        shape.insert(0, in1);
        if order == 'F':
            if len(args) == 0:
                shape.insert(0, in1); # row repeat
        return shape;
            