import { getTheme, mergeStyleSets, FontWeights, FontSizes } from '@fluentui/react';
// Themed styles for the example.
const theme = getTheme();
const contentStyles = mergeStyleSets({
    container: {
        display: 'flex',
        flexFlow: 'column nowrap',
        alignItems: 'stretch'
    },
    header: [
        theme.fonts.xLargePlus,
        {
            flex: '1 1 auto',
            borderTop: `4px solid ${theme.palette.themePrimary}`,
            color: theme.palette.neutralPrimary,
            display: 'flex',
            fontSize: FontSizes.xLarge,
            alignItems: 'center',
            fontWeight: FontWeights.semibold,
            padding: '12px 12px 14px 24px'
        }
    ],
    body: {
        flex: '4 4 auto',
        padding: '0 24px 24px 24px',
        overflowY: 'hidden',
        selectors: {
            p: {
                margin: '14px 0'
            },
            'p:first-child': {
                marginTop: 0
            },
            'p:last-child': {
                marginBottom: 0
            }
        }
    }
});

const iconButtonStyles = mergeStyleSets({
    root: {
        color: theme.palette.neutralPrimary,
        marginLeft: 'auto',
        marginTop: '4px',
        marginRight: '2px'
    },
    rootHovered: {
        color: theme.palette.neutralDark
    }
});

export { contentStyles, iconButtonStyles };
