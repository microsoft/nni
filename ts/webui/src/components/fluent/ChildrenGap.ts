import { IStackTokens, IStackStyles } from '@fluentui/react';

// name: gap + specific gap number

const gap10: IStackTokens = {
    childrenGap: 10
};

const gap15: IStackTokens = {
    childrenGap: 15
};

const stackStyle: IStackStyles = {
    root: {
        minWidth: 400,
        height: 56,
        display: 'flex',
        verticalAlign: 'center'
    }
};

export { gap10, gap15, stackStyle };
